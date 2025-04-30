# app/utils/result_viewer.py
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# StorageService 임포트
sys.path.append(str(Path(__file__).resolve().parents[2]))
from app.services.storage_service import StorageService

def format_json(data: Dict[str, Any], indent: int = 2) -> str:
    """JSON 데이터를 읽기 쉬운 형식으로 포매팅"""
    return json.dumps(data, ensure_ascii=False, indent=indent)

def print_colored(text: str, color: str = "reset") -> None:
    """텍스트에 색상 적용하여 출력"""
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        "reset": "\033[0m",
    }
    
    color_code = colors.get(color.lower(), colors["reset"])
    print(f"{color_code}{text}{colors['reset']}")

def truncate_text(text: str, max_length: int = 100) -> str:
    """텍스트가 너무 길 경우 축약"""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."

def list_comics(storage: StorageService, verbose: bool = False) -> None:
    """모든 만화 ID 목록 출력"""
    results = storage.list_results()
    
    if not results:
        print_colored("저장된 결과가 없습니다.", "yellow")
        return
    
    print_colored(f"총 {len(results)} 개의 만화 결과:", "cyan")
    
    for comic_id, agents in results.items():
        # 가장 최근 파일 찾기
        latest_file = None
        latest_time = 0
        
        for agent, files in agents.items():
            for file_info in files:
                path = file_info.get('path')
                if os.path.exists(path):
                    mtime = os.path.getmtime(path)
                    if mtime > latest_time:
                        latest_time = mtime
                        latest_file = path
        
        # 상세 정보 출력
        if verbose and latest_file:
            try:
                with open(latest_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    summary = data.get('result', {}).get('final_summary', '')
                    if not summary and isinstance(data.get('result'), dict):
                        # 다른 필드에서 요약 정보 찾기
                        for key in ['summary', 'description', 'message']:
                            if key in data.get('result', {}):
                                summary = data.get('result').get(key)
                                break
                    
                    if not summary and 'result' in data and isinstance(data['result'], str):
                        summary = data['result']
                        
                    print_colored(f"\n[{comic_id}]", "green")
                    if summary:
                        print_colored(f"요약: {truncate_text(summary)}", "white")
                    
                    # 에이전트 목록 출력
                    agents_list = ", ".join(list(agents.keys()))
                    print_colored(f"에이전트: {agents_list}", "blue")
            except Exception as e:
                print_colored(f"\n[{comic_id}] - 파일 읽기 오류: {e}", "red")
        else:
            # 간단히 ID만 출력
            agents_count = len(agents)
            print_colored(f"[{comic_id}] - {agents_count}개 에이전트 결과", "green")

def view_comic_details(storage: StorageService, comic_id: str, agent_name: Optional[str] = None) -> None:
    """특정 만화 ID의 상세 정보 출력"""
    if not comic_id:
        print_colored("만화 ID를 지정해야 합니다.", "red")
        return
    
    results = storage.list_results(comic_id=comic_id, agent_name=agent_name)
    
    if not results or comic_id not in results:
        print_colored(f"만화 ID [{comic_id}]에 대한 결과를 찾을 수 없습니다.", "red")
        return
    
    comic_data = results[comic_id]
    agents = list(comic_data.keys())
    
    print_colored(f"\n=== 만화 ID: {comic_id} ===", "cyan")
    print_colored(f"에이전트 목록: {', '.join(agents)}", "blue")
    
    for agent_name, files in comic_data.items():
        print_colored(f"\n## 에이전트: {agent_name} ({len(files)}개 파일)", "green")
        
        for idx, file_info in enumerate(files, 1):
            filename = file_info.get('filename')
            path = file_info.get('path')
            modified = file_info.get('modified')
            
            print_colored(f"\n[{idx}] {filename} ({modified})", "yellow")
            
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 결과 데이터 출력
                if 'result' in data:
                    result = data['result']
                    if isinstance(result, dict):
                        # 중요 필드만 출력
                        important_keys = [
                            'final_summary', 'summary', 'humor_texts', 'scenarios', 
                            'image_urls', 'error_message'
                        ]
                        
                        for key in important_keys:
                            if key in result:
                                value = result[key]
                                if isinstance(value, str):
                                    print_colored(f"{key}: {truncate_text(value)}", "white")
                                elif isinstance(value, list) and len(value) > 0:
                                    print_colored(f"{key}: {len(value)}개 항목", "white")
                                    for i, item in enumerate(value[:3], 1):
                                        if isinstance(item, str):
                                            print_colored(f"  {i}. {truncate_text(item)}", "white")
                                        elif isinstance(item, dict):
                                            print_colored(f"  {i}. {truncate_text(str(item))}", "white")
                    else:
                        print_colored(f"결과: {truncate_text(str(result))}", "white")
                
                # 오류 정보 출력
                if 'error' in data:
                    print_colored(f"오류: {data['error']}", "red")
                    if 'traceback' in data:
                        print_colored(f"스택 트레이스: {truncate_text(data['traceback'])}", "red")
                
                # 실행 시간 출력
                if 'execution_time' in data:
                    print_colored(f"실행 시간: {data['execution_time']:.2f}초", "blue")
                
            except Exception as e:
                print_colored(f"파일 읽기 오류: {e}", "red")

def view_full_result(storage: StorageService, comic_id: str, agent_name: str, file_index: int = 0) -> None:
    """특정 에이전트의 결과 전체 내용 출력"""
    results = storage.list_results(comic_id=comic_id, agent_name=agent_name)
    
    if not results or comic_id not in results:
        print_colored(f"만화 ID [{comic_id}]에 대한 결과를 찾을 수 없습니다.", "red")
        return
    
    if agent_name not in results[comic_id]:
        print_colored(f"에이전트 [{agent_name}]에 대한 결과를 찾을 수 없습니다.", "red")
        return
    
    files = results[comic_id][agent_name]
    
    if not files:
        print_colored(f"에이전트 [{agent_name}]에 대한 파일이 없습니다.", "red")
        return
    
    if file_index < 0 or file_index >= len(files):
        print_colored(f"파일 인덱스 [{file_index}]가 범위를 벗어납니다. (0-{len(files)-1})", "red")
        return
    
    file_info = files[file_index]
    path = file_info.get('path')
    
    print_colored(f"\n=== 만화 ID: {comic_id}, 에이전트: {agent_name}, 파일: {file_info.get('filename')} ===", "cyan")
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 전체 결과 출력
        print(format_json(data))
        
    except Exception as e:
        print_colored(f"파일 읽기 오류: {e}", "red")

def main():
    """메인 CLI 함수"""
    parser = argparse.ArgumentParser(description="만화 생성 결과 뷰어")
    
    subparsers = parser.add_subparsers(dest="command", help="명령어")
    
    # 목록 명령어
    list_parser = subparsers.add_parser("list", help="만화 결과 목록 출력")
    list_parser.add_argument("-v", "--verbose", action="store_true", help="상세 정보 출력")
    
    # 상세 보기 명령어
    view_parser = subparsers.add_parser("view", help="특정 만화 결과 상세 보기")
    view_parser.add_argument("comic_id", help="만화 ID")
    view_parser.add_argument("-a", "--agent", help="특정 에이전트 결과만 보기")
    
    # 전체 결과 보기 명령어
    full_parser = subparsers.add_parser("full", help="에이전트 결과 전체 내용 보기")
    full_parser.add_argument("comic_id", help="만화 ID")
    full_parser.add_argument("agent", help="에이전트 이름")
    full_parser.add_argument("-i", "--index", type=int, default=0, help="파일 인덱스 (기본값: 0)")
    
    # 결과 삭제 명령어
    clear_parser = subparsers.add_parser("clear", help="결과 삭제")
    clear_parser.add_argument("comic_id", nargs="?", help="삭제할 만화 ID (생략 시 모든 결과 삭제)")
    clear_parser.add_argument("-y", "--yes", action="store_true", help="확인 없이 삭제")
    
    args = parser.parse_args()
    
    # StorageService 인스턴스 생성
    storage = StorageService()
    
    if args.command == "list":
        list_comics(storage, args.verbose)
    
    elif args.command == "view":
        view_comic_details(storage, args.comic_id, args.agent)
    
    elif args.command == "full":
        view_full_result(storage, args.comic_id, args.agent, args.index)
    
    elif args.command == "clear":
        if not args.yes:
            if args.comic_id:
                confirm = input(f"만화 ID [{args.comic_id}]의 결과를 삭제하시겠습니까? (y/n): ")
            else:
                confirm = input("모든 결과를 삭제하시겠습니까? (y/n): ")
            
            if confirm.lower() != 'y':
                print_colored("삭제 취소됨", "yellow")
                return
        
        success = storage.clear_results(args.comic_id)
        
        if success:
            if args.comic_id:
                print_colored(f"만화 ID [{args.comic_id}]의 결과가 삭제되었습니다.", "green")
            else:
                print_colored("모든 결과가 삭제되었습니다.", "green")
        else:
            print_colored("결과 삭제 중 오류가 발생했습니다.", "red")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
