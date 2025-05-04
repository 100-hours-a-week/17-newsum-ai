# Workflow Progress Report (Phase 2: Scenario Generation)

**Generated:** `{{ timestamp }}`
**Trace ID:** `{{ trace_id }}`
**Comic ID:** `{{ comic_id }}`

---

## I. Chosen Scenario Overview

* **Chosen Idea Title:** `{{ chosen_title }}`
* **Scenario Prompt Hash (SHA256):** `{{ prompt_hash }}`

### Panel Summary

| Idea Title (Max 40) | Panel 1 Desc (Max 25) | Panel 2 Desc (Max 25) | Panel 3 Desc (Max 25) | Panel 4 Desc (Max 25) |
| :------------------ | :-------------------- | :-------------------- | :-------------------- | :-------------------- |
{% for row in mapping_rows %}
| {{ row.title }}     | {{ row.c1 }}          | {{ row.c2 }}          | {{ row.c3 }}          | {{ row.c4 }}          |
{% else %}
| No scenario data available. | - | - | - | - |
{% endfor %}

---

## II. Context Link Usage for Scenario

* **News Links Used:** `{{ link_usage.used_news }}` / `{{ link_usage.total_news }}`
* **Opinion Links Used:** `{{ link_usage.used_op }}` / `{{ link_usage.total_op }}`

{# Context links are those marked with status 'context_used' #}

---

## III. Scenario Quality Evaluation (LLM-based, 1-5 scale)

{% if quality_scores.consistency > 0 %} {# Assuming 0 means disabled/failed #}
* **Consistency:** {{ quality_scores.consistency }} / 5
* **Flow:** {{ quality_scores.flow }} / 5
* **Dialogue:** {{ quality_scores.dialogue }} / 5
{% else %}
* LLM-based scenario quality evaluation was disabled or failed.
{% endif %}

---

## IV. LLM Suggestions for Improvement

{% if suggestions and suggestions[0] != "Failed to generate suggestions." and suggestions[0] != "Evaluation disabled or failed." and suggestions[0] != "No specific suggestions generated." %}
{% for suggestion in suggestions %}
* {{ suggestion }}
{% endfor %}
{% else %}
* No improvement suggestions available (LLM generation disabled or failed).
{% endif %}