


#!/usr/bin/env python3
# course.py
"""
Advanced Course Enrollment Analyzer — V2
- Interactive missing-data handling (default = drop)
- Pandas & NumPy data manipulation previews
- Generates up to 10 visualization types (Matplotlib/Seaborn + Plotly for Gauge/Map)
- Saves outputs to analysis_outputs/ and runs an auto slideshow (custom timing & fullscreen option)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time, os, sys, webbrowser
sns.set(style='whitegrid')

def safe_input(prompt):
    try:
        return input(prompt)
    except EOFError:
        return ''

def try_import_plotly():
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        return go, px
    except Exception:
        return None, None

def save_fig_png(fig, path):
    try:
        fig.savefig(path, bbox_inches='tight')
    except Exception as e:
        print("Failed to save matplotlib figure:", e)

def save_plotly_fig(fig, path_html, path_png=None):
    try:
        fig.write_html(path_html)
        try:
            fig.write_image(path_png)
        except Exception:
            pass
    except Exception as e:
        print("Failed to save plotly figure:", e)

def detect_columns(df):
    cols = list(df.columns)
    def pick(cands):
        for c in cols:
            if c.lower() in cands:
                return c
        return None
    course = pick(['course','subject','program','class','course_name','course title'])
    enroll = pick(['enroll','enrollment','students','count','registrations','num_students','attendees','enrolled'])
    timec = pick(['month','date','year','time','period','timestamp','datetime','day'])
    # Try to find a datetime-like column if name-based pick didn't work
    if not timec:
        for c in cols:
            try:
                parsed = pd.to_datetime(df[c], errors='coerce', infer_datetime_format=True)
                # consider it a time column if a reasonable fraction parses successfully
                if parsed.notna().sum() >= max(1, int(len(df) * 0.3)):
                    timec = c
                    break
            except Exception:
                continue
    # If still not found, try year-like numeric column heuristic (e.g., 2019, 2020)
    if not timec:
        for c in cols:
            try:
                ser = pd.to_numeric(df[c], errors='coerce').dropna()
                if ser.empty:
                    continue
                within_years = ser.between(1900, 2100).sum()
                if within_years >= max(1, int(len(ser) * 0.5)):
                    timec = c
                    break
            except Exception:
                continue
    return course, enroll, timec

def handle_missing(df):
    miss = df.isnull().sum()
    miss = miss[miss>0]
    if miss.empty:
        print("No missing data detected.")
        return df, "none", 0, 0
    print("Detected missing data in columns:")
    for c,v in miss.items():
        print(f" - {c}: {v} missing")
    print("\nChoose how to handle missing data (press Enter to use DEFAULT = drop):")
    print("1. drop       - Remove rows with any missing values")
    print("2. fill_mean  - Replace missing numeric values with column mean")
    print("3. fill_zero  - Replace missing numeric values with 0")
    print("4. fill_custom - Replace all missing values with a custom value you provide")
    print("5. skip       - Do not modify (continue)")
    choice = safe_input("Enter choice (1/2/3/4/5) or press Enter: ").strip()
    before = len(df)
    if choice == '' or choice == '1' or choice.lower()=='drop':
        df_clean = df.dropna()
        method = 'drop'
    elif choice == '2' or choice.lower()=='fill_mean':
        numeric = df.select_dtypes(include='number').columns
        df_clean = df.copy()
        for c in numeric:
            df_clean[c] = df_clean[c].fillna(df_clean[c].mean())
        # For non-numeric, fill with mode if exists
        for c in df_clean.select_dtypes(exclude='number').columns:
            df_clean[c] = df_clean[c].fillna(df_clean[c].mode().iloc[0] if not df_clean[c].mode().empty else 'Unknown')
        method = 'fill_mean'
    elif choice == '3' or choice.lower()=='fill_zero':
        df_clean = df.fillna(0)
        method = 'fill_zero'
    elif choice == '4' or choice.lower()=='fill_custom':
        val = safe_input("Enter custom fill value (e.g., 0 or Unknown): ")
        df_clean = df.fillna(val)
        method = f'fill_custom({val})'
    elif choice == '5' or choice.lower()=='skip':
        df_clean = df.copy()
        method = 'skip'
    else:
        print("Unknown choice, defaulting to drop.")
        df_clean = df.dropna()
        method = 'drop'
    after = len(df_clean)
    print(f"Before cleaning: {before} rows -> After cleaning: {after} rows. Method: {method}")
    return df_clean, method, before, after

def summarize_manipulations(df, course_col, enroll_col):
    import numpy as np
    print("\n--- Data Manipulation Examples / Quick Summary ---")
    # Basic grouping
    grp = df.groupby(course_col)[enroll_col].agg(['sum','mean','max','min','count']).reset_index()
    print("\nTop rows of grouped summary (sum, mean, max, min, count):")
    print(grp.head().to_string(index=False, float_format='%.2f'))
    # Add engineered column: Enrollment_Level
    median = df[enroll_col].median()
    df['Enrollment_Level'] = np.where(df[enroll_col] > median, 'High', 'Low')
    print(f"\nAdded 'Enrollment_Level' based on median ({median:.2f}). Sample:")
    print(df[[course_col, enroll_col, 'Enrollment_Level']].head().to_string(index=False))
    # Normalization example
    df['Enroll_Normalized'] = (df[enroll_col] - df[enroll_col].min()) / (df[enroll_col].max() - df[enroll_col].min() + 1e-9)
    print("\nAdded 'Enroll_Normalized' (0-1 scaled) for visualization sizing.")
    return df, grp

# ------------------ ADDED: suggestion + single-window slideshow helpers ------------------
def suggest_visuals(df, course_col, enroll_col, time_col):
    """
    Inspect dataframe and suggest feasible visuals. Return a list of chosen visual keys.
    """
    feasible = []
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    has_numeric = len(numeric_cols) > 0
    has_two_numeric = len(numeric_cols) >= 2
    has_three_numeric = len(numeric_cols) >= 3
    has_time = bool(time_col and time_col in df.columns)
    lat_col = next((c for c in df.columns if 'lat' in c.lower()), None)
    lon_col = next((c for c in df.columns if 'lon' in c.lower() or 'long' in c.lower()), None)
    country_col = next((c for c in df.columns if c.lower() in ['country','country_name','location','city','state']), None)
    has_geo = bool((lat_col and lon_col) or country_col)

    if course_col and enroll_col:
        feasible.append(('bar', 'Bar Chart - total per course'))
        feasible.append(('pie', 'Pie Chart - course share (if <=15 categories)'))
        feasible.append(('donut', 'Donut Chart - course share (if <=15 categories)'))
        feasible.append(('box', 'Box Plot - enrollment distribution by course'))
        # add course-based line/area for course_id / course_title supportive view
        feasible.append(('line_course', 'Line Chart - total enrollments by course (ordered)'))
        feasible.append(('area_course', 'Area Chart - total enrollments by course (filled)'))
    if has_numeric:
        feasible.append(('hist', 'Histogram - enrollment distribution'))
    if has_time:
        feasible.append(('line', 'Line Chart - trends over time'))
        feasible.append(('area', 'Area Chart - trends over time'))
    if has_two_numeric:
        feasible.append(('scatter', 'Scatter Plot - numeric relationships'))
    if has_three_numeric:
        feasible.append(('bubble', 'Bubble Chart - 3rd numeric for size'))
    if len(numeric_cols) >= 2:
        feasible.append(('heatmap', 'Heatmap - numeric correlation'))
    if has_geo:
        feasible.append(('map', 'Map - geographic distribution'))
    feasible.append(('gauge', 'Gauge - top course share (%) (requires plotly)'))

    print("\nSuggested visualizations based on dataset columns:")
    for i,(k,desc) in enumerate(feasible, start=1):
        print(f" {i}. {desc} [{k}]")
    print("\nPress Enter to choose ALL suggested visuals, or enter comma-separated numbers (e.g. 1,3,5) to pick:")
    choice = safe_input("Your choice: ").strip()
    if choice == '':
        return [k for k,_ in feasible]
    selected = []
    try:
        indices = [int(x.strip()) for x in choice.split(',') if x.strip()]
        for idx in indices:
            if 1 <= idx <= len(feasible):
                selected.append(feasible[idx-1][0])
    except Exception:
        print("Invalid selection; defaulting to all suggested visuals.")
        return [k for k,_ in feasible]
    if not selected:
        print("No valid selection made; defaulting to all suggested visuals.")
        return [k for k,_ in feasible]
    return selected

def run_slideshow(charts, sleep_sec=4.0, use_fullscreen=False, open_html=True):
    """
    Display saved PNG images one by one in the same matplotlib window.
    If open_html is True, open any chart HTML files (Plotly gauge/map) in browser tabs before starting the slideshow.
    charts: list of tuples (title, png_path, html_path)
    """
    # Open HTML files (Plotly) in browser tabs first (one-time)
    if open_html:
        opened = set()
        for title, img_path, html_path in charts:
            if html_path and os.path.exists(html_path):
                abspath = os.path.abspath(html_path)
                if abspath not in opened:
                    try:
                        print(f"Opening in browser: {title} -> {html_path}")
                        webbrowser.open_new_tab('file://' + abspath)
                        opened.add(abspath)
                        time.sleep(0.3)  # small delay to let browser open the tab
                    except Exception:
                        pass

    # Then run the single-window image slideshow
    plt.ion()
    fig, ax = plt.subplots(figsize=(10,6))
    if use_fullscreen:
        try:
            mng = plt.get_current_fig_manager()
            mng.full_screen_toggle()
        except Exception:
            pass

    for title, img_path, html_path in charts:
        # Prefer showing PNG if available
        if img_path and os.path.exists(img_path):
            try:
                ax.clear()
                img = plt.imread(img_path)
                ax.imshow(img)
                ax.axis('off')
                ax.set_title(title)
                fig.canvas.draw()
                plt.pause(sleep_sec)
            except Exception:
                print("Unable to display image for:", title)
                time.sleep(min(2, sleep_sec))
        else:
            # If no PNG but HTML exists, we already opened it in the browser — notify and skip
            if html_path and os.path.exists(html_path):
                print(f"{title} is open in your browser (HTML): {html_path}")
            else:
                print("No visual available for:", title)
            time.sleep(1.0)

    plt.ioff()
    plt.close(fig)
# ------------------ END added helpers ------------------

def main():
    print("📂 Course Enrollment Analyzer")
    mode = safe_input("Type 'upload' to provide a CSV path, or paste the path directly: ").strip()
    if mode.lower() == 'upload' or mode == '':
        path = safe_input("👉 Paste full CSV file path: ").strip().strip('\"')
    else:
        path = mode.strip().strip('\"')
    if not path:
        print("No path provided. Exiting.")
        sys.exit(0)
    if not os.path.exists(path):
        print("File not found:", path); sys.exit(1)

    df = pd.read_csv(path)
    print("\n✅ File loaded. Columns:", list(df.columns))
    print(df.head().to_string(index=False))

    # detect columns
    course_col, enroll_col, time_col = detect_columns(df)
    print(f"Auto-detected columns -> Course: {course_col}, Enrollments: {enroll_col}, Time: {time_col}")
    if not course_col:
        course_col = safe_input("Enter course column name exactly: ").strip()
    if not enroll_col:
        enroll_col = safe_input("Enter enrollment column name exactly: ").strip()
    if course_col not in df.columns or enroll_col not in df.columns:
        print("Required columns missing. Exiting."); sys.exit(1)

    # Handle missing data interactively (default drop)
    df_clean, method, before, after = handle_missing(df)

    # Convert enroll to numeric and drop NA in enroll
    df_clean[enroll_col] = pd.to_numeric(df_clean[enroll_col], errors='coerce')
    df_clean = df_clean.dropna(subset=[enroll_col])

    # ------------------ ADD: suggest visuals BEFORE heavy plotting ------------------
    selected_visuals = suggest_visuals(df_clean, course_col, enroll_col, time_col)
    print("Selected visuals:", selected_visuals)
    # ------------------ END suggest visuals ------------------

    # Summarize manipulations & feature engineering
    df_transformed, grouped_summary = summarize_manipulations(df_clean, course_col, enroll_col)

    # Aggregation for visuals
    total = df_transformed.groupby(course_col)[enroll_col].sum().reset_index().sort_values(enroll_col, ascending=False)
    total['Percentage'] = (total[enroll_col] / total[enroll_col].sum()) * 100

    # Create output directory based on CSV file name
    csv_basename = os.path.splitext(os.path.basename(path))[0]
    out_dir = f'{csv_basename}_analysis_outputs'
    os.makedirs(out_dir, exist_ok=True)

    # slideshow options
    use_fullscreen = safe_input("Enable fullscreen slide preview? (y/N): ").strip().lower() == 'y'
    try:
        sleep_sec = float(safe_input("Slide duration seconds (default 4): ").strip() or "4")
    except ValueError:
        sleep_sec = 4.0

    # reference image
    ref_img = 'visual_types_reference.png'
    charts = []
    if os.path.exists(ref_img):
        charts.append(("Visualization Types Reference", ref_img, None))

    # Basic plots - only produce if selected
    # Bar
    if 'bar' in selected_visuals:
        fig, ax = plt.subplots(figsize=(10,6))
        sns.barplot(x=course_col, y=enroll_col, data=total, ax=ax)
        ax.set_title('Total Enrollments per Course'); ax.set_xlabel('Course'); ax.set_ylabel('Enrollments')
        plt.xticks(rotation=45, ha='right')
        bar_p = os.path.join(out_dir, 'bar_total_enrollments.png'); save_fig_png(fig, bar_p); plt.close(fig)
        charts.append(("Bar Chart - Total Enrollments", bar_p, None))

    # Pie
    if 'pie' in selected_visuals and len(total) <= 15:
        fig, ax = plt.subplots(figsize=(7,7))
        ax.pie(total[enroll_col], labels=total[course_col], autopct='%1.1f%%', startangle=140)
        ax.set_title('Course Share (Pie Chart)')
        pie_p = os.path.join(out_dir, 'pie_course_share.png'); save_fig_png(fig, pie_p); plt.close(fig)
        charts.append(("Pie Chart - Course Share", pie_p, None))

    # Donut
    if 'donut' in selected_visuals and len(total) <= 15:
        fig, ax = plt.subplots(figsize=(7,7))
        wedges, texts, autotexts = ax.pie(total[enroll_col], labels=total[course_col], autopct='%1.1f%%', startangle=140, wedgeprops=dict(width=0.3))
        ax.set_title('Course Share (Donut Chart)')
        donut_p = os.path.join(out_dir, 'donut_course_share.png'); save_fig_png(fig, donut_p); plt.close(fig)
        charts.append(("Donut Chart - Course Share", donut_p, None))

    # Box Plot
    if 'box' in selected_visuals:
        fig, ax = plt.subplots(figsize=(10,6))
        sns.boxplot(x=course_col, y=enroll_col, data=df_transformed, hue=course_col, palette='Set3', legend=False, ax=ax)
        ax.set_title('Enrollment Distribution by Course (Box Plot)')
        ax.set_xlabel('Course'); ax.set_ylabel('Enrollments')
        plt.xticks(rotation=45, ha='right')
        box_p = os.path.join(out_dir, 'box_enrollment_by_course.png'); save_fig_png(fig, box_p); plt.close(fig)
        charts.append(("Box Plot - Enrollment by Course", box_p, None))

    # Histogram
    if 'hist' in selected_visuals:
        fig, ax = plt.subplots(figsize=(9,5))
        sns.histplot(df_transformed[enroll_col], bins=min(20, max(5, len(df_transformed)//2)), kde=True, ax=ax)
        ax.set_title('Enrollment Distribution (Histogram)')
        hist_p = os.path.join(out_dir, 'hist_enrollment.png'); save_fig_png(fig, hist_p); plt.close(fig)
        charts.append(("Histogram - Enrollment Distribution", hist_p, None))

    # Course-based line & area (new)
    if ('line_course' in selected_visuals or 'area_course' in selected_visuals) and not total.empty:
        try:
            # Ensure a stable order (by total enrollments)
            total_sorted = total.sort_values(by=enroll_col, ascending=False).reset_index(drop=True)
            x = total_sorted[course_col].astype(str)
            y = total_sorted[enroll_col].values

            if 'line_course' in selected_visuals:
                fig, ax = plt.subplots(figsize=(12,6))
                sns.lineplot(x=x, y=y, marker='o', sort=False, ax=ax)
                ax.set_title('Total Enrollments by Course (Line)')
                ax.set_xlabel('Course'); ax.set_ylabel('Enrollments')
                plt.xticks(rotation=45, ha='right')
                line_course_p = os.path.join(out_dir, 'line_course_total.png'); save_fig_png(fig, line_course_p); plt.close(fig)
                charts.append(("Line Chart - Enrollments by Course", line_course_p, None))

            if 'area_course' in selected_visuals:
                fig, ax = plt.subplots(figsize=(12,6))
                ax.plot(x, y, marker='o')
                ax.fill_between(range(len(x)), y, alpha=0.3)
                ax.set_xticks(range(len(x))); ax.set_xticklabels(x, rotation=45, ha='right')
                ax.set_title('Total Enrollments by Course (Area)')
                ax.set_xlabel('Course'); ax.set_ylabel('Enrollments')
                area_course_p = os.path.join(out_dir, 'area_course_total.png'); save_fig_png(fig, area_course_p); plt.close(fig)
                charts.append(("Area Chart - Enrollments by Course", area_course_p, None))
        except Exception as e:
            print("Course-based line/area charts skipped:", e)

    # Time-based line & area
    if ('line' in selected_visuals or 'area' in selected_visuals) and time_col and time_col in df_transformed.columns:
        try:
            df_time = df_transformed.copy(); df_time[time_col] = pd.to_datetime(df_time[time_col], errors='coerce')
            agg = df_time.groupby([time_col, course_col])[enroll_col].sum().reset_index()
            if 'line' in selected_visuals:
                fig, ax = plt.subplots(figsize=(10,6))
                sns.lineplot(x=time_col, y=enroll_col, hue=course_col, data=agg, marker='o', ax=ax)
                ax.set_title('Enrollment Trends Over Time (Line)'); plt.xticks(rotation=45)
                line_p = os.path.join(out_dir, 'line_trends.png'); save_fig_png(fig, line_p); plt.close(fig)
                charts.append(("Line Chart - Trends Over Time", line_p, None))
            if 'area' in selected_visuals:
                pivot = agg.pivot(index=time_col, columns=course_col, values=enroll_col).fillna(0)
                fig, ax = plt.subplots(figsize=(10,6)); pivot.plot.area(ax=ax); ax.set_title('Enrollment Trends Over Time (Area)')
                area_p = os.path.join(out_dir, 'area_trends.png'); save_fig_png(fig, area_p); plt.close(fig)
                charts.append(("Area Chart - Trends Over Time", area_p, None))
        except Exception as e:
            print("Time charts skipped:", e)

    # Scatter & Bubble & Heatmap
    numeric_cols = df_transformed.select_dtypes(include='number').columns.tolist()
    if 'scatter' in selected_visuals and len(numeric_cols) >= 2:
        other_nums = [c for c in numeric_cols if c != enroll_col]
        x_col = other_nums[0] if other_nums else enroll_col
        y_col = other_nums[1] if len(other_nums) > 1 else enroll_col
        # Add categories based on dataset (e.g., course_col)
        fig, ax = plt.subplots(figsize=(8,6)); sns.scatterplot(x=x_col, y=y_col, data=df_transformed, hue=course_col, ax=ax)
        ax.set_title(f'Scatter Plot - {x_col} vs {y_col} by {course_col}')
        scatter_p = os.path.join(out_dir, 'scatter_numeric.png'); save_fig_png(fig, scatter_p); plt.close(fig)
        charts.append((f"Scatter Plot - {x_col} vs {y_col}", scatter_p, None))

    if 'bubble' in selected_visuals and len(numeric_cols) >= 3:
        size_col = numeric_cols[2]
        fig, ax = plt.subplots(figsize=(8,6))
        sns.scatterplot(x=x_col, y=y_col, size=size_col, hue=course_col, data=df_transformed, ax=ax, alpha=0.6)
        ax.set_title('Bubble Chart')
        bubble_p = os.path.join(out_dir, 'bubble_chart.png'); save_fig_png(fig, bubble_p); plt.close(fig)
        charts.append(("Bubble Chart", bubble_p, None))

    if 'heatmap' in selected_visuals and len(numeric_cols) >= 2:
        try:
            corr = df_transformed[numeric_cols].corr(); fig, ax = plt.subplots(figsize=(8,6)); sns.heatmap(corr, annot=True, fmt='.2f', ax=ax)
            heat_p = os.path.join(out_dir, 'heatmap_corr.png'); save_fig_png(fig, heat_p); plt.close(fig); charts.append(("Heatmap - Numeric Correlation", heat_p, None))
        except Exception:
            pass

    # Plotly-based Gauge & Map if available and selected
    go, px = try_import_plotly()
    lat_col = next((c for c in df_transformed.columns if 'lat' in c.lower()), None)
    lon_col = next((c for c in df_transformed.columns if 'lon' in c.lower() or 'long' in c.lower()), None)
    country_col = next((c for c in df_transformed.columns if c.lower() in ['country','country_name','location','city','state']), None)

    if go and px:
        if 'gauge' in selected_visuals:
            try:
                top_pct = float(total['Percentage'].iloc[0]) if not total.empty else 0.0
                gauge_fig = go.Figure(go.Indicator(mode="gauge+number+delta", value=top_pct, title={'text': "Top Course Share (%)"}, delta={'reference': total['Percentage'].mean() if not total.empty else 0}, gauge={'axis': {'range': [None, 100]}}))
                g_html = os.path.join(out_dir, 'gauge_top_course.html'); g_png = os.path.join(out_dir, 'gauge_top_course.png')
                save_plotly_fig(gauge_fig, g_html, g_png)
                charts.append(("Gauge - Top Course Share", g_png if os.path.exists(g_png) else None, g_html))
            except Exception as e:
                print("Gauge creation failed:", e)
        if 'map' in selected_visuals and ((lat_col and lon_col) or country_col):
            try:
                if lat_col and lon_col:
                    map_fig = px.scatter_geo(df_transformed, lat=lat_col, lon=lon_col, hover_name=course_col, size=enroll_col, projection='natural earth')
                else:
                    map_fig = px.choropleth(df_transformed, locations=country_col, locationmode='country names', color=enroll_col, hover_name=course_col)
                map_html = os.path.join(out_dir, 'map.html'); map_png = os.path.join(out_dir, 'map.png')
                save_plotly_fig(map_fig, map_html, map_png)
                charts.append(("Map - Geographic Distribution", map_png if os.path.exists(map_png) else None, map_html))
            except Exception as e:
                print("Map creation failed:", e)
    else:
        if 'gauge' in selected_visuals or 'map' in selected_visuals:
            print("Plotly not available: Gauge/Map may still be saved as HTML but PNG export requires plotly + kaleido.")

    # Save summary CSV
    total.to_csv(os.path.join(out_dir, 'total_enrollments_by_course_v2.csv'), index=False)
    grouped_summary.to_csv(os.path.join(out_dir, 'grouped_summary_v2.csv'), index=False)

    # Run single-window slideshow
    print(f"\n📊 Starting Auto-Slideshow: {sleep_sec}s per slide. Fullscreen={'Yes' if use_fullscreen else 'No'}")
    run_slideshow(charts, sleep_sec=sleep_sec, use_fullscreen=use_fullscreen, open_html=False)

    # After slideshow, ask to display Gauge and Map
    if 'gauge' in selected_visuals or 'map' in selected_visuals:
        display_gauge_map = safe_input("Do you want to display Gauge and Map charts in web browser now? (y/N): ").strip().lower() == 'y'
        if display_gauge_map:
            for chart in charts:
                if 'Gauge' in chart[0] or 'Map' in chart[0]:
                    if chart[2] and os.path.exists(chart[2]):
                        try:
                            webbrowser.open_new_tab('file://' + os.path.abspath(chart[2]))
                        except Exception:
                            pass

    print("\n✅ Analysis & slideshow complete. Outputs saved in:", os.path.abspath(out_dir))

if __name__ == '__main__':
    main()