import time
import sys
from src.utils.agent_graph import AgentWorkflow
from src.utils.cache_memory import ExactMatchCache, SemanticCache
from src.utils.mcp_client import MCPClient

def measure():
    flow = AgentWorkflow(ExactMatchCache(), SemanticCache(), MCPClient())
    t0 = time.time()
    res = flow.run('Plot a bar chart showing Apple quarterly revenue for the four quarters of 2024.')
    t1 = time.time()
    with open('logs_time.txt', 'w', encoding='utf-8') as f:
        f.write(str(res.get('run_log', [])))
        f.write('\nTotal time: ' + str(t1 - t0))
        
if __name__ == '__main__':
    measure()
