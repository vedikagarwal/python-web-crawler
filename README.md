# python-web-crawler
Multi-threaded Python web crawler with keyword extraction, SQLite storage, and performance monitoring

## Requirements

### Python Dependencies
Create a requirements.txt file with the following contents:
- requests>=2.31.0
- beautifulsoup4>=4.12.0
- matplotlib>=3.7.0
- pandas>=2.0.0
- lxml>=4.9.0

### System Requirements
- Python 3.8 or higher
- Internet connection
- At least 1GB free disk space for crawled data

## Quick Start

### Clone the repository

```bash
git clone https://github.com/vedikagarwal/python-web-crawler.git
cd python-web-crawler
```

### Run the workflow

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the web_crawler:
```bash
python web_crawler.py --seed-urls https://cc.gatech.edu --max-urls 1000 --threads 5 --delay 1.0
```
