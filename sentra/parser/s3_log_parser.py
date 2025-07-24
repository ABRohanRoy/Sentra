import re
from datetime import datetime

def parse_s3_log_line(line):
    pattern = re.compile(
        r'(?P<request_id>[a-f0-9]{64})\s+(?P<bucket>[^\s]+)\s+\[(?P<timestamp>[^\]]+)\]\s+'
        r'(?P<ip>\d+\.\d+\.\d+\.\d+)\s+-\s+(?P<aws_id>[A-Z0-9]+)\s+'
        r'(?P<action>WEBSITE\.[A-Z_\.]+)\s+(?P<object>[^\s]+)\s+"(?P<method>[A-Z]+)\s+(?P<endpoint>[^"]+)\s+HTTP/[0-9\.]+"\s+'
        r'(?P<status>\d+)'
    )

    match = pattern.search(line)
    if not match:
        return None

    data = match.groupdict()

    try:
        data["timestamp"] = datetime.strptime(data["timestamp"], "%d/%b/%Y:%H:%M:%S %z").isoformat()
    except Exception:
        pass

    return data


def parse_log_file(filepath):
    """
    Reads and parses an S3 access log file.
    Returns a list of parsed log entries.
    """
    parsed_logs = []
    with open(filepath, 'r') as file:
        for line in file:
            parsed = parse_s3_log_line(line)
            if parsed:
                parsed_logs.append(parsed)
    return parsed_logs


if __name__ == "__main__":
    import json

    input_path = "data/sample_logs/s3_access_sample.log"
    parsed = parse_log_file(input_path)
    print(json.dumps(parsed[:3], indent=2))
