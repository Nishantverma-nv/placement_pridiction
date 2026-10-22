"""PlacementPredict — entry point.

    python main.py      -> serves the web app at http://127.0.0.1:5000
    python scripts/pipeline.py -> rebuilds data, models, charts and reports
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.app import app

if __name__ == "__main__":
    print("PlacementPredict running on http://127.0.0.1:5000  (Ctrl+C to stop)")
    app.run(debug=True, host="127.0.0.1", port=5000)
