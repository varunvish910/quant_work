#!/bin/bash
# Cleanup and organize root directory

echo "Organizing files..."

# Move analysis outputs
mv SPY_Options_30Day_Complete_Analysis.png analysis_outputs/oct_2025_spy_options/
mv SPY_OPTIONS_ANALYSIS_SUMMARY_OCT2025.md analysis_outputs/oct_2025_spy_options/
mv spy_options_current_snapshot.csv analysis_outputs/oct_2025_spy_options/

# Move scripts
mv analyze_spy_flatfiles_complete.py scripts/options_analysis/
mv analyze_spx_research_with_spy.py scripts/options_analysis/
mv download_polygon_flatfiles_s3.py scripts/options_analysis/
mv download_and_match_quotes.py scripts/options_analysis/
mv download_spy_options_historical.py scripts/options_analysis/
mv list_available_flatfiles.py scripts/options_analysis/
mv analyze_spy_options_anomalies.py scripts/options_analysis/
mv analyze_spy_smh_options_realtime.py scripts/options_analysis/
mv download_historical_spy_api.py scripts/options_analysis/

# Archive old visualizations and analysis
mv October_2025_Detailed_Analysis.png archive/oct_2025/
mv DETAILED_ANALYSIS_OCT2025.md archive/oct_2025/
mv SPY_ANALYSIS_REPORT.md archive/oct_2025/
mv SPY_Options_Anomaly_Analysis.png archive/oct_2025/
mv SPY_Risk_Dashboard.png archive/oct_2025/
mv SPY_Risk_Summary.png archive/oct_2025/
mv VIX_Term_Structure_Analysis.png archive/oct_2025/
mv VIX_Term_Structure_Daily.png archive/oct_2025/
mv visualize_october_focus.py archive/oct_2025/
mv visualize_risk.py archive/oct_2025/
mv visualize_risk_simple.py archive/oct_2025/
mv visualize_vix_term_structure.py archive/oct_2025/

echo "✅ Organization complete!"
echo ""
echo "Structure:"
echo "  analysis_outputs/oct_2025_spy_options/ - Final analysis outputs"
echo "  scripts/options_analysis/              - Analysis scripts"  
echo "  archive/oct_2025/                      - Old/interim work"
