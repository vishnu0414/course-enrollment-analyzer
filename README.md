# Course Enrollment Analyzer

## Overview
The **Course Enrollment Analyzer** is a Python script designed to analyze course enrollment data from CSV files. It provides interactive data cleaning, generates various visualizations (up to 10 types), and runs an automated slideshow of the results. This tool is ideal for educators, administrators, or analysts looking to gain insights into enrollment patterns, trends, and distributions.

## Features
- **Interactive Missing Data Handling**: Choose how to handle missing values (drop, fill with mean/zero/custom value, or skip).
- **Automatic Column Detection**: Intelligently detects course, enrollment, and time columns based on common naming conventions or data patterns.
- **Data Manipulation Previews**: Includes examples of grouping, feature engineering (e.g., enrollment levels, normalization), and summary statistics.
- **Visualization Options**: Supports up to 10 visualization types, including:
  - Bar charts, pie/donut charts, box plots
  - Histograms, line/area charts (time-based and course-based)
  - Scatter plots, bubble charts, heatmaps
  - Gauge charts (top course share) and geographic maps (if applicable)
- **Slideshow Mode**: Automatically displays generated charts in a single window with customizable timing and fullscreen option.
- **Output Saving**: Saves all visualizations, summaries, and data to a dedicated output directory.
- **Web Integration**: Opens interactive Plotly charts (gauge/map) in the browser for enhanced viewing.

## Requirements
- Python 3.x
- Libraries: pandas, numpy, matplotlib, seaborn
- Optional: plotly (for gauge and map visualizations), kaleido (for PNG export of Plotly charts)

## Installation
1. Ensure Python 3.x is installed on your system.
2. Install required libraries:
   ```
   pip install pandas numpy matplotlib seaborn
   ```
3. For optional Plotly features:
   ```
   pip install plotly kaleido
   ```

## Usage
1. Run the script:
   ```
   python course.py
   ```
2. Provide the path to your CSV file when prompted (or type 'upload' to paste the path).
3. The script will:
   - Load and preview the data.
   - Auto-detect relevant columns (course, enrollment, time).
   - Prompt for missing data handling.
   - Suggest and allow selection of visualizations.
   - Generate and save outputs to `{csv_filename}_analysis_outputs/`.
   - Run an automated slideshow of the charts.
4. After the slideshow, you can choose to view interactive charts in your web browser.

## Input Data Format
- CSV file with columns representing courses, enrollments, and optionally time/geographic data.
- The script attempts to auto-detect columns, but you can manually specify if needed.
- Enrollment data should be numeric.

## Output
- **Directory**: `{csv_filename}_analysis_outputs/` containing:
  - PNG images of all generated charts.
  - HTML files for interactive Plotly visualizations.
  - CSV files with summarized data (total enrollments, grouped summaries).
- **Slideshow**: Automatic display of charts in a matplotlib window.
- **Browser**: Interactive gauge and map charts opened in new tabs.

## Examples
- Analyze enrollment trends over time with line charts.
- Visualize course popularity with bar and pie charts.
- Explore correlations between numeric variables with heatmaps and scatter plots.
- Display geographic distribution if location data is available.

## License
This project is open-source. Feel free to modify and distribute.

## Contributing
Contributions are welcome! Please submit issues or pull requests on the project repository.

## Version
V2 - Includes enhanced visualization suggestions, single-window slideshow, and improved data handling.
