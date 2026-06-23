"""Analyze AGNT tool usage from the database."""
import sqlite3, json, os

db_path = os.path.join(os.environ.get('APPDATA', ''), 'AGNT', 'Data', 'agnt.db')
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
c = conn.cursor()

# Check schema of key tables
for table in ['tools', 'node_executions', 'agent_tool_executions', 'installed_plugin_assets', 'workflow_executions']:
    try:
        c.execute("PRAGMA table_info(" + table + ")")
        cols = [r[1] for r in c.fetchall()]
        print(table + ": " + str(cols))
    except Exception as e:
        print(table + " error: " + str(e))

# Get tool usage from node_executions
print("\n=== NODE EXECUTIONS ===")
try:
    c.execute("SELECT * FROM node_executions LIMIT 3")
    for r in c.fetchall():
        print(dict(r))
except Exception as e:
    print("Error:", e)

# Get agent tool executions
print("\n=== AGENT TOOL EXECUTIONS ===")
try:
    c.execute("SELECT * FROM agent_tool_executions LIMIT 3")
    for r in c.fetchall():
        print(dict(r))
except Exception as e:
    print("Error:", e)

# Get tools
print("\n=== TOOLS ===")
try:
    c.execute("SELECT * FROM tools LIMIT 10")
    for r in c.fetchall():
        print(dict(r))
except Exception as e:
    print("Error:", e)

# Count executions by workflow
print("\n=== EXECUTION COUNTS BY WORKFLOW ===")
try:
    c.execute("SELECT workflow_name, COUNT(*) as cnt, SUM(CASE WHEN status='success' THEN 1 ELSE 0 END) as successes FROM workflow_executions GROUP BY workflow_name ORDER BY cnt DESC LIMIT 20")
    for r in c.fetchall():
        print("  " + str(r[0])[:40] + " | " + str(r[1]) + " execs | " + str(r[2]) + " success")
except Exception as e:
    print("Error:", e)

conn.close()
