import os
import markdown
import jinja2
from datetime import datetime
from app.config.settings import settings

def parse_sections_from_markdown(md_text: str):
    """
    마크다운 텍스트에서 '##' 헤더 기준으로 섹션을 분리합니다.
    각 섹션은 {"title": ..., "content": ...} 형태로 반환됩니다.
    """
    import re
    sections = []
    current = None
    lines = md_text.splitlines()
    for line in lines:
        if line.startswith('## '):
            if current:
                sections.append(current)
            current = {"title": line[3:].strip(), "content": ""}
        else:
            if current:
                current["content"] += line + "\n"
    if current:
        sections.append(current)
    # 만약 맨 앞에 # 제목이 있으면, 첫 섹션의 title로 사용
    if sections and lines and lines[0].startswith('# '):
        sections[0]["title"] = lines[0][2:].strip()
    return sections

def render_report_html_from_markdown(work_id: str, title: str, md_text: str):
    # 섹션 분리
    sections = parse_sections_from_markdown(md_text)
    # 마크다운 → HTML 변환
    for section in sections:
        section["html_content"] = markdown.markdown(section["content"], extensions=["extra", "tables", "fenced_code", "nl2br"])
    # Jinja2 템플릿 로드 (settings에서 경로 지정)
    template_path = settings.REPORT_TEMPLATE_PATH
    template_dir, template_file = os.path.split(template_path)
    jinja_env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(template_dir),
        autoescape=True
    )
    template = jinja_env.get_template(template_file)
    html = template.render(
        title=title,
        work_id=work_id,
        created_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        sections=sections
    )
    return html

def save_report_html(work_id: str, html: str):
    output_dir = settings.REPORT_HTML_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{work_id}.html")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(html)
    return file_path 