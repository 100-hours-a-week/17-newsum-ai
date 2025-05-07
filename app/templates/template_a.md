# Workflow Progress Report (Phase 1: Analysis & Evaluation)

**Generated:** `{{ timestamp }}`
**Trace ID:** `{{ trace_id }}`
**Comic ID:** `{{ comic_id }}`

---

## I. Summary & Evaluation

### Final Summary Preview (Max 300 chars)
`{{ final_summary_preview }}`


### Evaluation Metrics

| Metric                | Score     | Threshold (Example) | Status   |
| :-------------------- | :-------- | :------------------ | :------- |
| Topic Coverage        | {{ topic_match_percent }} | > 70%             | {% if (evaluation_metrics.topic_coverage * 100) > 70 %}✅ Pass{% else %}❗ Fail{% endif %} |
| ROUGE-L (F1)          | {{ rouge_l_score }}         | > 0.35            | {% if evaluation_metrics.rouge_l > 0.35 %}✅ Pass{% else %}❗ Fail{% endif %} |
| BERTScore (F1)        | {{ bertscore_f1 }}        | > 0.88            | {% if evaluation_metrics.bert_score > 0.88 %}✅ Pass{% else %}❗ Fail{% endif %} |
{# Note: Thresholds are examples, adjust as needed #}

### Decision & Next Step

* **Evaluation Decision:** `{{ decision }}`
* **Next Step:** {{ next_step }}

---

## II. Trend Analysis

### Top {{ trend_scores_top_n | length }} Keywords

| Rank | Keyword             | Score (0-100) |
| :--- | :------------------ | :------------ |
{% for trend in trend_scores_top_n %}
| {{ trend.rank }} | `{{ trend.keyword }}` | {{ trend.score }}    |
{% else %}
| -    | No trend data found. | -             |
{% endfor %}

---

## III. Data Sources & Link Status

### Fact URLs (News Articles, etc.)

| #    | URL Preview (Max 60 chars)                   | Status | Purpose / Keyword (Max 40 chars) |
| :--- | :------------------------------------------- | :----- | :------------------------------- |
{% for item in fact_urls_info %}
| {{ item.index }} | `{{ item.url }}`                             | {{ item.status_symbol }} | `{{ item.purpose }}`           |
{% else %}
| -    | No fact URLs collected or processed.         | -      | -                                |
{% endfor %}

### Opinion URLs (Social Media, Blogs, etc.)

| #    | URL Preview (Max 60 chars)                   | Status | Purpose / Keyword (Max 40 chars) |
| :--- | :------------------------------------------- | :----- | :------------------------------- |
{% for item in opinion_urls_info %}
| {{ item.index }} | `{{ item.url }}`                             | {{ item.status_symbol }} | `{{ item.purpose }}`           |
{% else %}
| -    | No opinion URLs collected or processed.      | -      | -                                |
{% endfor %}

**Status Key:** `✓` Processed, `C` Context Used, `-` Collected/Tracked, `!` Failed, `?` Unknown

---

## IV. Workflow Performance (Up to Node 12)

### Node Processing Times

| Node        | Time (s) |
| :---------- | :------- |
{% for node, time in node_processing_times.items() %}
| {{ node }} | {{ time }}    |
{% else %}
| No timing data available. | -        |
{% endfor %}

**Total Elapsed Time (Nodes 1-12):** `{{ total_elapsed_time }} seconds`

