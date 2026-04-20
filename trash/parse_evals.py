
import csv
import os

def parse_csv(path):
    results = []
    if not os.path.exists(path):
        return results
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)
    return results

def read_utf16le(path):
    if not os.path.exists(path):
        return ""
    try:
        with open(path, 'r', encoding='utf-16le') as f:
            return f.read()
    except Exception as e:
        return str(e)

csv_path = r"c:\Users\Administrator\Downloads\Omiya-fixed\Omiya-main\runs\lstm_cpu\evaluation_summary.csv"
eval_out_path = r"c:\Users\Administrator\Downloads\Omiya-fixed\Omiya-main\eval_out.txt"

csv_data = parse_csv(csv_path)
eval_out_data = read_utf16le(eval_out_path)

print("--- CSV Data Summary ---")
# Group by 100% progress to see final win rates of each block
finals = [row for row in csv_data if row.get('progress_pct') == '100']
for i, row in enumerate(finals):
    print(f"Run {i+1}: Team A Win Rate: {row.get('team_a_win_rate')}% (Episodes: {row.get('episodes_completed')})")

print("\n--- eval_out.txt Snapshot ---")
# Just show the last few lines of eval_out.txt as it might contain the most recent manual eval
lines = eval_out_data.strip().split('\n')
for line in lines[-20:]:
    print(line)
