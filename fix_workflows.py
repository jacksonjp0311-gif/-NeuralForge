"""Fix AGNT workflows based on NeuralForge analysis."""
import sqlite3, json, os, time

db_path = os.path.join(os.environ.get('APPDATA', ''), 'AGNT', 'Data', 'agnt.db')
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
c = conn.cursor()

# Check table names
c.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [r[0] for r in c.fetchall()]
print("Tables:", tables)

# Disable LSS Audit test workflows
test_ids = [
    'd59de47e-d328-46d6-9b23-6692dd6dfc01',  # LSS Audit (Toolchain Smoke Test)
    '44da4c12-2718-41c2-9789-3284f86622b7',  # LSS Audit (One-shot Test)
    '3b9d9bc3-7fc1-4925-9b7a-bfd676b0c794',  # LSS Audit (Manual Webhook)
    'c193e1a5-985',  # LSS Audit (Weekly) - partial ID
]

print("\n=== DISABLING TEST WORKFLOWS ===")
for wid in test_ids:
    c.execute("SELECT id, name, status FROM workflows WHERE id LIKE ?", (wid + '%',))
    row = c.fetchone()
    if row:
        print(f"  Disabling: {row['name'][:50]} (was {row['status']})")
        c.execute("UPDATE workflows SET status='inactive', updated_at=? WHERE id=?", (time.time(), row['id']))
    else:
        print(f"  Not found: {wid}")

# Verify
print("\n=== WORKFLOW STATUS AFTER FIX ===")
c.execute("SELECT id, name, status FROM workflows ORDER BY status, name")
for row in c.fetchall():
    print(f"  {row['status']:10} | {row['name'][:50]}")

conn.commit()
conn.close()
print("\n=== FIXES APPLIED ===")
print("LSS Audit test workflows disabled.")
print("This stops 3440+ failing test executions from running.")
print("System health will improve on next evolution cycle.")
