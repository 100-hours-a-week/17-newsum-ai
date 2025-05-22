# ai/app/nodes/n05_hitl_review_node.py
from typing import Dict, Any, Optional
from datetime import datetime, timezone
import asyncio
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from bs4 import BeautifulSoup
import re

from app.workflows.state_v2 import WorkflowState
from app.utils.logger import get_logger
from app.services.llm_service import LLMService

logger = get_logger(__name__)
console = Console()

class N05HITLReviewNode:
    """보고서 생성 후 사용자 검토를 위한 HITL 노드"""

    def __init__(self, llm_service: LLMService):
        self.name = "n05_hitl_review"
        self.llm_service = llm_service
        self.MAX_MODIFICATION_ATTEMPTS = 3  # 최대 수정 시도 횟수

    def _extract_summary(self, html_content: str) -> str:
        """HTML 보고서에서 요약 정보를 추출합니다."""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # 제목 추출
            title = soup.find('h1')
            title_text = title.get_text() if title else "제목 없음"
            
            # 첫 번째 단락 추출
            first_p = soup.find('p')
            first_p_text = first_p.get_text() if first_p else "내용 없음"
            
            # 주요 섹션 제목들 추출
            sections = soup.find_all(['h2', 'h3'])
            section_titles = [section.get_text() for section in sections[:3]]  # 최대 3개 섹션
            
            # 요약본 생성
            summary = f"""
제목: {title_text}

요약: {first_p_text[:200]}...

주요 섹션:
{chr(10).join(f'- {title}' for title in section_titles)}
"""
            return summary
        except Exception as e:
            logger.error(f"요약 추출 중 오류 발생: {str(e)}")
            return "요약을 생성할 수 없습니다."

    async def _display_report(self, report_content: str) -> None:
        """보고서 내용을 CLI에 표시합니다."""
        # 요약본 생성
        summary = self._extract_summary(report_content)
        
        console.print("\n[bold blue]=== 보고서 요약 ===[/bold blue]")
        console.print(Panel(summary, title="보고서 요약", border_style="blue"))
        
        # 전체 내용 보기 옵션
        if Confirm.ask("\n전체 보고서 내용을 보시겠습니까?"):
            console.print("\n[bold blue]=== 전체 보고서 ===[/bold blue]")
            console.print(Panel(report_content, title="전체 보고서", border_style="blue"))
        
        console.print("\n")

    async def _get_user_feedback(self) -> tuple[str, str]:
        """사용자로부터 피드백을 받습니다."""
        console.print("[bold yellow]보고서에 대한 피드백을 입력해주세요:[/bold yellow]")
        feedback = Prompt.ask("피드백")
        
        action = Prompt.ask(
            "다음 작업을 선택하세요",
            choices=["approve", "reject", "modify"],
            default="approve"
        )
        
        return action, feedback

    async def _handle_modification(self, report_sec, meta_sec, llm_service) -> bool:
        """보고서 수정 요청을 처리합니다."""
        console.print("[bold yellow]수정이 필요한 부분을 설명해주세요(slm에게 prompt로 들어갈 내용입니다):[/bold yellow]")
        modification_request = Prompt.ask("수정 요청")
        
        # LLM을 사용하여 보고서 수정
        modification_prompt = f"""
Please revise the following report based on the user's feedback.

Original Report:
{report_sec.report_content}

User's Modification Request:
{modification_request}

Return the revised report in **HTML format only**.
"""
        
        try:
            response = await llm_service.generate_text(modification_prompt)
            # LLM 응답에서 generated_text 추출
            if isinstance(response, dict) and 'generated_text' in response:
                modified_report = response['generated_text']
            else:
                modified_report = str(response)
                
            report_sec.report_content = modified_report
            
            # 수정 이력 기록
            report_sec.hitl_revision_history.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "action": "modify",
                "feedback": modification_request,
                "report_content": modified_report
            })
            
            console.print("[bold green]보고서가 수정되었습니다.[/bold green]")
            return True
            
        except Exception as e:
            logger.error(f"보고서 수정 중 오류 발생: {str(e)}")
            meta_sec.error_message = f"보고서 수정 실패: {str(e)}"
            report_sec.hitl_status = "error"
            meta_sec.current_stage = "ERROR"
            return False

    async def run(self, state: WorkflowState) -> Dict[str, Any]:
        """
        HITL 리뷰 프로세스를 실행합니다.
        state_v2 구조에 맞게 Section 객체를 직접 사용하며, 주석도 최대한 유지합니다.
        """
        meta = state.meta
        report_sec = state.report
        node_name = self.name
        meta.current_stage = node_name
        meta.error_message = None
        meta.error_log = list(meta.error_log or [])
        report_sec.hitl_status = "pending"
        report_sec.hitl_last_updated = datetime.now(timezone.utc).isoformat()
        # 보고서가 없는 경우 에러 처리
        if not report_sec.report_content:
            meta.error_message = "HITL 리뷰를 위한 보고서가 없습니다."
            meta.current_stage = "ERROR"
            logger.error(f"[{node_name}] {meta.error_message}")
            return {"report": report_sec.model_dump(), "meta": meta.model_dump()}
        # HITL 리뷰 이력 초기화
        report_sec.hitl_revision_history = [{
            "timestamp": report_sec.hitl_last_updated,
            "action": "initial_review",
            "report_content": report_sec.report_content
        }]
        # 보고서 표시
        await self._display_report(report_sec.report_content)
        modification_count = 0
        while True:
            # 사용자 피드백 받기
            action, feedback = await self._get_user_feedback()
            # 피드백 저장
            report_sec.hitl_feedback = feedback
            report_sec.hitl_status = action
            report_sec.hitl_last_updated = datetime.now(timezone.utc).isoformat()
            # 수정 이력 기록
            report_sec.hitl_revision_history.append({
                "timestamp": report_sec.hitl_last_updated,
                "action": action,
                "feedback": feedback
            })
            if action == "approve":
                console.print("[bold green]보고서가 승인되었습니다.[/bold green]")
                break
            elif action == "reject":
                console.print("[bold red]보고서가 거부되었습니다.[/bold red]")
                meta.error_message = f"사용자에 의해 보고서가 거부됨: {feedback}"
                report_sec.hitl_status = "rejected"
                meta.current_stage = "intentionally_terminated"
                meta.workflow_status = "intentionally_terminated"
                break
            elif action == "modify":
                if modification_count >= self.MAX_MODIFICATION_ATTEMPTS:
                    console.print("[bold red]최대 수정 횟수(3회)를 초과했습니다.[/bold red]")
                    meta.error_message = "최대 수정 횟수를 초과했습니다."
                    report_sec.hitl_status = "max_modifications_reached"
                    meta.current_stage = "intentionally_terminated"
                    meta.workflow_status = "intentionally_terminated"
                    break
                success = await self._handle_modification(report_sec, meta, self.llm_service)
                if not success and meta.error_message:
                    break
                modification_count += 1
                console.print(f"[bold yellow]남은 수정 횟수: {self.MAX_MODIFICATION_ATTEMPTS - modification_count}회[/bold yellow]")
                # 수정된 보고서 표시
                await self._display_report(report_sec.report_content)
                continue
        logger.info(f"[{node_name}] HITL 리뷰 프로세스 완료")
        return {"report": report_sec.model_dump(), "meta": meta.model_dump()} 