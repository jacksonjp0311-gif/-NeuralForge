"""Query AGNT database for real execution data and feed into NeuralForge."""
import sqlite3
import os
import json
import sys
import io
from datetime import datetime

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def find_db():
    paths = [
        os.path.join(os.environ.get('APPDATA', ''), 'AGNT', 'Data', 'agnt.db'),
        os.path.join(os.environ.get('USERPROFILE', ''), 'Documents', 'AGNT_Data', 'agnt.db'),
    ]
    for p in paths:
        if os.path.exists(p):
            return p
    return None

def main():
    db_path = find_db()
    if not db_path:
        print('ERROR: No AGNT database found')
        sys.exit(1)
    
    print(f'Database: {db_path}')
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    cur.execute('SELECT COUNT(*) FROM workflow_executions')
    total = cur.fetchone()[0]
    print(f'Total workflow executions: {total}')
    
    if total == 0:
        print('No executions found.')
        conn.close()
        return
    
    # Node execution stats
    cur.execute('SELECT status, COUNT(*) FROM node_executions GROUP BY status')
    print('\nNode status breakdown:')
    for r in cur.fetchall():
        print(f'  {r[0]}: {r[1]}')
    
    # Per-workflow stats
    cur.execute('''SELECT workflow_name, COUNT(*), 
               SUM(CASE WHEN status='completed' THEN 1 ELSE 0 END)
               FROM workflow_executions GROUP BY workflow_name ORDER BY COUNT(*) DESC LIMIT 15''')
    print('\nPer-workflow stats:')
    for r in cur.fetchall():
        success_rate = (r[2]/r[1]*100) if r[1] > 0 else 0
        name = (r[0] or 'Unknown')[:50]
        print(f'  {name}: {r[1]} runs, {success_rate:.0f}% success')
    
    # Export all executions
    cur.execute('''
        SELECT we.id, we.workflow_id, we.workflow_name, we.status, 
               we.start_time, we.end_time, we.credits_used
        FROM workflow_executions we
        ORDER BY start_time DESC
    ''')
    all_rows = cur.fetchall()
    
    executions = []
    for r in all_rows:
        cur2 = conn.cursor()
        cur2.execute('''
            SELECT node_id, status, start_time, end_time, error, credits_used,
                   input_tokens, output_tokens
            FROM node_executions
            WHERE execution_id = ?
            ORDER BY start_time
        ''', (r[0],))
        nodes = cur2.fetchall()
        
        duration_ms = 0
        if r[4] and r[5]:
            try:
                start = datetime.fromisoformat(r[4].replace('Z', '+00:00').replace('+00:00', ''))
                end = datetime.fromisoformat(r[5].replace('Z', '+00:00').replace('+00:00', ''))
                duration_ms = (end - start).total_seconds() * 1000
            except:
                pass
        
        error_nodes = [n for n in nodes if n[1] == 'error']
        success = r[3] == 'completed' and len(error_nodes) == 0
        
        executions.append({
            'workflow_id': r[0],
            'workflow_name': r[2] or 'Unknown',
            'status': r[3],
            'success': success,
            'duration_ms': duration_ms,
            'step_count': len(nodes),
            'retry_count': 0,
            'error_type': error_nodes[0][4] if error_nodes else 'none',
            'recovery_action': '',
            'recovery_success': False,
            'params': {},
            'prompt': '',
            'response': '',
            'timestamp': r[4] or '',
            '_data_source': 'real_executions',
        })
    
    # Save to JSON
    output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'agnt_executions.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(executions, f, indent=2, default=str)
    
    print(f'\nExported {len(executions)} executions to: {output_path}')
    
    # Ingest into NeuralForge event log
    event_log = os.path.join(os.path.dirname(__file__), '..', 'cold_storage', 'neuralforge', 'execution_events.jsonl')
    os.makedirs(os.path.dirname(event_log), exist_ok=True)
    
    new_count = 0
    for e in executions:
        with open(event_log, 'a', encoding='utf-8') as f:
            f.write(json.dumps(e, ensure_ascii=False) + '\n')
            new_count += 1
    
    print(f'Ingested {new_count} events into: {event_log}')
    
    # Summary stats
    successful = sum(1 for e in executions if e['success'])
    print(f'\nSummary:')
    print(f'  Total: {len(executions)}')
    print(f'  Successful: {successful}')
    print(f'  Failed: {len(executions) - successful}')
    print(f'  Success rate: {successful/len(executions)*100:.1f}%')
    
    conn.close()

if __name__ == '__main__':
    main()
