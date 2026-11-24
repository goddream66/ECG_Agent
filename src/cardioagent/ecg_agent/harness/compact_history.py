"""A reversible prompt view of repeated measurement rows. The evidence ledger stays verbatim."""
import json


TABLE_GUIDE = (' Read each measurement table in row order. Fields in shared apply to every row. '
               'Match the values in each row to columns from left to right. IDs, scores and times are unchanged.')


def table_view(data):
    output = data.copy()
    for key in ('earlier_measurements_for_comparison_only', 'rhythm_by_time'):
        rows = data.get(key)
        if not isinstance(rows, list) or len(rows) < 3:
            continue
        first = rows[0]
        if not isinstance(first, dict) or not all(isinstance(row, dict) and row.keys() == first.keys() for row in rows):
            continue
        # Even if all values match, keep attribution, time and numeric values beside each observation.
        shared = {field: first[field] for field in ('tool','status','warnings','direction_allowed')
                  if field in first and all(row[field] == first[field] for row in rows)}
        columns = [field for field in first if field not in shared]
        table = dict(shared=shared, columns=columns, rows=[[row[field] for field in columns] for row in rows])
        if len(json.dumps(table)) + len(TABLE_GUIDE) < len(json.dumps(rows)):
            output[key] = table
    return output
