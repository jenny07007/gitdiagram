from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from app.services.github_service import GitHubService
from app.prompts import (
    SYSTEM_FIRST_PROMPT,
    SYSTEM_SECOND_PROMPT,
    SYSTEM_THIRD_PROMPT,
    ADDITIONAL_SYSTEM_INSTRUCTIONS_PROMPT,
)
from anthropic._exceptions import RateLimitError
from pydantic import BaseModel
from functools import lru_cache
import re
import json
import asyncio
from app.services.llm_service_factory import LLMServiceFactory

# from app.services.claude_service import ClaudeService
# from app.core.limiter import limiter

load_dotenv()

router = APIRouter(prefix="/generate", tags=["Claude"])

# Initialize LLM service using factory
llm_service = LLMServiceFactory.create_service(
    "openai-o3")  # or "openrouter-o3" or "claude"


@lru_cache(maxsize=100)
def get_cached_github_data(username: str, repo: str, github_pat: str | None = None):
    current_github_service = GitHubService(pat=github_pat)
    default_branch = current_github_service.get_default_branch(username, repo)
    if not default_branch:
        default_branch = "main"
    file_tree = current_github_service.get_github_file_paths_as_list(
        username, repo)
    readme = current_github_service.get_github_readme(username, repo)
    return {"default_branch": default_branch, "file_tree": file_tree, "readme": readme}


class ApiRequest(BaseModel):
    username: str
    repo: str
    instructions: str = ""
    api_key: str | None = None
    github_pat: str | None = None


@router.post("")
# @limiter.limit("1/minute;5/day")
async def generate(request: Request, body: ApiRequest):
    try:
        if len(body.instructions) > 1000:
            return {"error": "Instructions exceed maximum length of 1000 characters"}
        if body.repo in ["fastapi", "streamlit", "flask", "api-analytics", "monkeytype"]:
            return {"error": "Example repos cannot be regenerated"}

        github_data = get_cached_github_data(
            body.username, body.repo, body.github_pat)
        default_branch = github_data["default_branch"]
        file_tree = github_data["file_tree"]
        readme = github_data["readme"]

        combined_content = f"{file_tree}\n{readme}"
        token_count = llm_service.count_tokens(combined_content)

        if 50000 < token_count < 195000 and not body.api_key:
            return {
                "error": f"File tree and README combined exceeds token limit (50,000). Current size: {token_count} tokens. This GitHub repository is too large for my wallet, but you can continue by providing your own OpenRouter API key.",
                "token_count": token_count,
                "requires_api_key": True,
            }
        elif token_count > 195000:
            return {
                "error": f"Repository is too large (>195k tokens) for analysis. OpenAI o3-mini's max context length is 200k tokens. Current size: {token_count} tokens."
            }

        first_system_prompt = SYSTEM_FIRST_PROMPT
        third_system_prompt = SYSTEM_THIRD_PROMPT
        if body.instructions:
            first_system_prompt += "\n" + ADDITIONAL_SYSTEM_INSTRUCTIONS_PROMPT
            third_system_prompt += "\n" + ADDITIONAL_SYSTEM_INSTRUCTIONS_PROMPT

        explanation = llm_service.call_api(
            system_prompt=first_system_prompt,
            data={"file_tree": file_tree, "readme": readme,
                  "instructions": body.instructions},
            api_key=body.api_key,
            reasoning_effort="medium",
        )
        if "BAD_INSTRUCTIONS" in explanation:
            return {"error": "Invalid or unclear instructions provided"}

        full_second_response = llm_service.call_api(
            system_prompt=SYSTEM_SECOND_PROMPT,
            data={"explanation": explanation, "file_tree": file_tree},
            api_key=body.api_key,
        )
        start_tag = "<component_mapping>"
        end_tag = "</component_mapping>"
        component_mapping_text = full_second_response[
            full_second_response.find(start_tag): full_second_response.find(end_tag)
        ]
        mermaid_code = llm_service.call_api(
            system_prompt=third_system_prompt,
            data={
                "explanation": explanation,
                "component_mapping": component_mapping_text,
                "instructions": body.instructions,
            },
            api_key=body.api_key,
            reasoning_effort="medium",
        )

        # check for and remove code block tags
        mermaid_code = mermaid_code.replace(
            "```mermaid", "").replace("```", "")

        if "BAD_INSTRUCTIONS" in mermaid_code:
            return {"error": "Invalid or unclear instructions provided"}

        processed_diagram = process_click_events(
            mermaid_code, body.username, body.repo, default_branch)
        return {"diagram": processed_diagram, "explanation": explanation}
    except RateLimitError as e:
        raise HTTPException(
            status_code=429,
            detail="Service is currently experiencing high demand. Please try again in a few minutes.",
        )
    except Exception as e:
        return {"error": str(e)}


@router.post("/cost")
async def get_generation_cost(request: Request, body: ApiRequest):
    try:
        github_data = get_cached_github_data(
            body.username, body.repo, body.github_pat)
        file_tree = github_data["file_tree"]
        readme = github_data["readme"]
        file_tree_tokens = llm_service.count_tokens(file_tree)
        readme_tokens = llm_service.count_tokens(readme)
        input_cost = ((file_tree_tokens * 2 + readme_tokens) +
                      3000) * 0.0000011
        output_cost = 8000 * 0.0000044
        estimated_cost = input_cost + output_cost
        cost_string = f"${estimated_cost:.2f} USD"
        return {"cost": cost_string}
    except Exception as e:
        return {"error": str(e)}


def process_click_events(diagram: str, username: str, repo: str, branch: str) -> str:
    def replace_path(match):
        path = match.group(2).strip("\"'")
        is_file = "." in path.split("/")[-1]
        base_url = f"https://github.com/{username}/{repo}"
        path_type = "blob" if is_file else "tree"
        full_url = f"{base_url}/{path_type}/{branch}/{path}"
        return f'click {match.group(1)} "{full_url}"'
    click_pattern = r'click ([^\s"]+)\s+"([^"]+)"'
    return re.sub(click_pattern, replace_path, diagram)


@router.post("/stream")
async def generate_stream(request: Request, body: ApiRequest):
    try:
        if len(body.instructions) > 1000:
            return {"error": "Instructions exceed maximum length of 1000 characters"}
        if body.repo in ["fastapi", "streamlit", "flask", "api-analytics", "monkeytype"]:
            return {"error": "Example repos cannot be regenerated"}

        async def event_generator():
            try:
                MAX_SAFE_LENGTH = 3500  # Safe length for JSON strings

                def truncate_safely(text: str, max_length: int = MAX_SAFE_LENGTH) -> str:
                    """Truncate text to a safe length and ensure it ends properly"""
                    if len(text) <= max_length:
                        return text
                    truncated = text[:max_length]
                    # Add ellipsis to indicate truncation
                    return truncated + "..."

                github_data = get_cached_github_data(
                    body.username, body.repo, body.github_pat)
                default_branch = github_data["default_branch"]
                file_tree = github_data["file_tree"]
                readme = github_data["readme"]

                yield f"data: {json.dumps({'status': 'started', 'message': 'Starting generation process ...'})}\n\n"
                await asyncio.sleep(0.1)

                combined_content = f"{file_tree}\n{readme}"
                token_count = llm_service.count_tokens(combined_content)
                if 50000 < token_count < 195000 and not body.api_key:
                    yield f"data: {json.dumps({'error': f'File tree and README combined exceeds token limit (50,000). Current size: {token_count} tokens. This GitHub repository is too large for my wallet, but you can continue by providing your own OpenRouter API key.'})}\n\n"
                    return
                elif token_count > 195000:
                    yield f"data: {json.dumps({'error': f'Repository is too large (>195k tokens) for analysis. OpenAI o3-mini\'s max context length is 200k tokens. Current size: {token_count} tokens.'})}\n\n"
                    return

                first_system_prompt = SYSTEM_FIRST_PROMPT
                third_system_prompt = SYSTEM_THIRD_PROMPT
                if body.instructions:
                    first_system_prompt += "\n" + ADDITIONAL_SYSTEM_INSTRUCTIONS_PROMPT
                    third_system_prompt += "\n" + ADDITIONAL_SYSTEM_INSTRUCTIONS_PROMPT

                yield f"data: {json.dumps({'status': 'explanation_sent', 'message': 'Sending explanation request to o3-mini...'})}\n\n"
                await asyncio.sleep(0.1)
                yield f"data: {json.dumps({'status': 'explanation', 'message': 'Analyzing repository structure...'})}\n\n"
                explanation = ""
                async for chunk in llm_service.call_api_stream(
                    system_prompt=first_system_prompt,
                    data={"file_tree": file_tree, "readme": readme,
                          "instructions": body.instructions},
                    api_key=body.api_key,
                    reasoning_effort="medium",
                ):
                    explanation += chunk
                    yield f"data: {json.dumps({'status': 'explanation_chunk', 'chunk': chunk})}\n\n"

                if "BAD_INSTRUCTIONS" in explanation:
                    yield f"data: {json.dumps({'error': 'Invalid or unclear instructions provided'})}\n\n"
                    return

                yield f"data: {json.dumps({'status': 'mapping_sent', 'message': 'Sending component mapping request to o3-mini...'})}\n\n"
                await asyncio.sleep(0.1)
                yield f"data: {json.dumps({'status': 'mapping', 'message': 'Creating component mapping...'})}\n\n"
                full_second_response = ""
                async for chunk in llm_service.call_api_stream(
                    system_prompt=SYSTEM_SECOND_PROMPT,
                    data={"explanation": explanation, "file_tree": file_tree},
                    api_key=body.api_key,
                    reasoning_effort="medium",
                ):
                    full_second_response += chunk
                    yield f"data: {json.dumps({'status': 'mapping_chunk', 'chunk': chunk})}\n\n"

                start_tag = "<component_mapping>"
                end_tag = "</component_mapping>"
                component_mapping_text = full_second_response[
                    full_second_response.find(start_tag): full_second_response.find(end_tag)
                ]

                yield f"data: {json.dumps({'status': 'diagram_sent', 'message': 'Sending diagram generation request to o3-mini...'})}\n\n"
                await asyncio.sleep(0.1)
                yield f"data: {json.dumps({'status': 'diagram', 'message': 'Generating diagram...'})}\n\n"
                mermaid_code = ""
                async for chunk in llm_service.call_api_stream(
                    system_prompt=third_system_prompt,
                    data={"explanation": explanation, "component_mapping": component_mapping_text,
                          "instructions": body.instructions},
                    api_key=body.api_key,
                    reasoning_effort="medium",
                ):
                    mermaid_code += chunk
                    yield f"data: {json.dumps({'status': 'diagram_chunk', 'chunk': chunk})}\n\n"

                # Process final diagram
                mermaid_code = mermaid_code.replace(
                    "```mermaid", "").replace("```", "")
                if "BAD_INSTRUCTIONS" in mermaid_code:
                    yield f"data: {json.dumps({'error': 'Invalid or unclear instructions provided'})}\n\n"
                    return

                processed_diagram = process_click_events(
                    mermaid_code, body.username, body.repo, default_branch)

                print("=== Generated Mermaid Diagram ===")
                print(processed_diagram)
                print("================================")

                # Send diagram first
                yield f"data: {json.dumps({
                    'status': 'complete_diagram',
                    'diagram': processed_diagram
                })}\n\n"

                # Split explanation into smaller chunks
                CHUNK_SIZE = 3000
                explanation_chunks = [explanation[i:i + CHUNK_SIZE]
                                      for i in range(0, len(explanation), CHUNK_SIZE)]

                # Send explanation chunks
                for i, chunk in enumerate(explanation_chunks):
                    yield f"data: {json.dumps({
                        'status': 'complete_explanation',
                        'explanation_chunk': chunk,
                        'is_last': i == len(explanation_chunks) - 1
                    })}\n\n"

                # Finally send mapping
                yield f"data: {json.dumps({
                    'status': 'complete_mapping',
                    'mapping': component_mapping_text
                })}\n\n"

                # Send completion signal
                yield f"data: {json.dumps({'status': 'complete'})}\n\n"

            except Exception as e:
                print("=== Stream Error ===")
                print(str(e))
                print("==================")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "X-Accel-Buffering": "no",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
    except Exception as e:
        return {"error": str(e)}
