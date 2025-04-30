# app/agents/scenariowriter_agent.py
import logging
from typing import Dict, Optional, Any, List
from app.workflows.state import ComicState
from app.services.llm_server_client import call_llm_api

logger = logging.getLogger(__name__)

class ScenarioWriterAgent:
    """
    Agent that creates 4-panel comic scenarios based on humor points and sentiment analysis results
    """
    
    async def run(self, state: ComicState) -> Dict[str, Optional[Any]]:
        """
        Uses state.final_summary, state.humor_texts, and state.public_sentiment 
        to generate a 4-panel comic scenario.
        """
        logger.info("--- [Scenario Writer Agent] Starting execution ---")
        updates: Dict[str, Optional[Any]] = {}
        
        # Check required inputs
        if not state.final_summary:
            logger.warning("[ScenarioWriter] No summary available for scenario creation.")
            updates["scenarios"] = []
            return updates
            
        humor_texts = state.humor_texts or []
        public_sentiment = state.public_sentiment
        
        # Identify sentiment information
        sentiment_info = "No sentiment data available"
        dominant_emotion = "neutral"
        
        if public_sentiment:
            try:
                dominant_sentiment = max(public_sentiment["sentiment"].items(), key=lambda x: x[1])[0]
                dominant_emotion = max(public_sentiment["emotions"].items(), key=lambda x: x[1])[0]
                sentiment_info = f"Dominant sentiment: {dominant_sentiment}, Dominant emotion: {dominant_emotion}"
                logger.info(f"[ScenarioWriter] Sentiment analysis info: {sentiment_info}")
            except Exception as e:
                logger.error(f"[ScenarioWriter] Error processing sentiment data: {e}")
        
        # Humor points string
        humor_points = "\n".join([f"- {point}" for point in humor_texts]) if humor_texts else "No humor points available."
        
        # Construct prompt
        prompt = f"""Create a 4-panel comic scenario based on the following information:

News Summary:
---
{state.final_summary[:2000]}
---

Humor Points:
---
{humor_points}
---

Public Sentiment Analysis:
---
{sentiment_info}
---

Scenario Guidelines:
1. Create a 4-panel comic scenario.
2. Each panel should include a scene description and character dialogue.
3. Consider the public's dominant emotion '{dominant_emotion}' and use an appropriate tone.
4. Utilize the provided humor points, but avoid mocking victims/vulnerable groups and focus on systemic absurdities or ironies.
5. Ensure panels flow naturally, with a clear message delivered in the final panel.

Respond in exactly this format:
[Panel 1]
Scene: (description of panel 1)
Dialogue: (dialogue in panel 1)

[Panel 2]
Scene: (description of panel 2)
Dialogue: (dialogue in panel 2)

[Panel 3]
Scene: (description of panel 3)
Dialogue: (dialogue in panel 3)

[Panel 4]
Scene: (description of panel 4)
Dialogue: (dialogue in panel 4)
"""

        try:
            # LLM API call
            response = await call_llm_api(prompt, max_tokens=800, temperature=0.7)
            logger.info("[ScenarioWriter] Successfully received LLM response.")
            
            # Parse results
            scenarios = []
            current_cut = {}
            current_section = None
            
            for line in response.splitlines():
                line = line.strip()
                
                if line.startswith('[Panel') and line.endswith(']'):
                    if current_cut and 'description' in current_cut and 'dialogue' in current_cut:
                        scenarios.append(current_cut)
                    current_cut = {'description': '', 'dialogue': ''}
                    current_section = None
                    
                elif line.lower().startswith('scene:'):
                    current_section = 'description'
                    current_cut['description'] = line[6:].strip()
                    
                elif line.lower().startswith('dialogue:'):
                    current_section = 'dialogue'
                    current_cut['dialogue'] = line[9:].strip()
                    
                elif current_section:
                    current_cut[current_section] += ' ' + line
            
            # Add the last panel
            if current_cut and 'description' in current_cut and 'dialogue' in current_cut:
                scenarios.append(current_cut)
                
            # Ensure we have exactly 4 panels
            while len(scenarios) < 4:
                scenarios.append({"description": f"Additional scene {len(scenarios)+1}", "dialogue": "..."})
            
            # Use at most 4 panels
            scenarios = scenarios[:4]
            
            logger.info(f"[ScenarioWriter] Successfully created {len(scenarios)} scenario panels")
            updates["scenarios"] = scenarios
            updates["error_message"] = None
            
        except Exception as e:
            logger.error(f"[ScenarioWriter] Error during scenario creation: {e}")
            # Provide basic scenarios on error
            basic_scenarios = [
                {"description": "Basic news-related scene", "dialogue": "Dialogue about the news content"},
                {"description": "Situation development", "dialogue": "Reaction to the issue"},
                {"description": "Conflict or turning point", "dialogue": "Attempt to resolve the situation"},
                {"description": "Conclusion", "dialogue": "Closing remarks"}
            ]
            updates["scenarios"] = basic_scenarios
            updates["error_message"] = f"Failed to generate scenarios: {str(e)}"
            
        logger.info("--- [Scenario Writer Agent] Execution complete ---")
        return updates