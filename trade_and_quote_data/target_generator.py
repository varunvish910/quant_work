#!/usr/bin/env python3
"""
Target Generator - Creates binary targets for correction prediction
"""

import pandas as pd
import numpy as np
from typing import List, Dict
from pathlib import Path
import json
from datetime import datetime

class TargetGenerator:
    """Creates binary target labels for correction prediction"""
    
    def create_prediction_targets(self, price_data: pd.DataFrame, correction_events: List[Dict]) -> pd.DataFrame:
        """Create binary target labels: 1 for days 1-3 before corrections, 0 otherwise"""
        if not correction_events:
            return pd.DataFrame(columns=['date', 'target', 'days_to_correction', 'correction_magnitude'])
        
        # Initialize targets dataframe
        targets_df = price_data[['date']].copy()
        targets_df['target'] = 0
        targets_df['days_to_correction'] = np.nan
        targets_df['correction_magnitude'] = np.nan
        targets_df['correction_peak_date'] = pd.NaT
        
        # Sort corrections by magnitude (largest first) to handle overlaps
        sorted_corrections = sorted(correction_events, key=lambda x: x['magnitude'], reverse=True)
        
        for correction in sorted_corrections:
            peak_date = correction['peak_date']
            magnitude = correction['magnitude']
            
            # Find the peak date in our dataframe
            peak_mask = targets_df['date'] == peak_date
            if not peak_mask.any():
                continue
                
            peak_idx = targets_df[peak_mask].index[0]
            
            # Mark days 1-3 before the correction as targets
            for days_before in range(1, 4):
                target_idx = peak_idx - days_before
                if target_idx >= 0:
                    # Only set target if not already set (prioritize larger corrections)
                    if targets_df.loc[target_idx, 'target'] == 0:
                        targets_df.loc[target_idx, 'target'] = 1
                        targets_df.loc[target_idx, 'days_to_correction'] = days_before
                        targets_df.loc[target_idx, 'correction_magnitude'] = magnitude
                        targets_df.loc[target_idx, 'correction_peak_date'] = peak_date
        
        # Add additional features
        targets_df['is_target'] = targets_df['target'] == 1
        targets_df['correction_category'] = pd.cut(
            targets_df['correction_magnitude'], 
            bins=[0, 0.05, 0.08, 0.12, 1.0], 
            labels=['Minor (4-5%)', 'Moderate (5-8%)', 'Major (8-12%)', 'Severe (12%+)'],
            include_lowest=True
        )
        
        print(f"🎯 Created {targets_df['target'].sum()} prediction targets")
        print(f"📊 Target distribution: {targets_df['target'].value_counts().to_dict()}")
        
        return targets_df
    
    def validate_targets(self, targets_df: pd.DataFrame) -> Dict:
        """Validate target distribution and timing"""
        if targets_df.empty:
            return {"error": "No targets to validate"}
        
        validation = {}
        
        # Basic statistics
        total_days = len(targets_df)
        target_days = targets_df['target'].sum()
        non_target_days = total_days - target_days
        
        validation['total_days'] = total_days
        validation['target_days'] = int(target_days)
        validation['non_target_days'] = int(non_target_days)
        validation['target_ratio'] = target_days / total_days if total_days > 0 else 0
        
        # Target distribution by days before correction
        if target_days > 0:
            days_dist = targets_df[targets_df['target'] == 1]['days_to_correction'].value_counts().sort_index()
            validation['days_to_correction_distribution'] = days_dist.to_dict()
            
            # Average days before correction
            validation['avg_days_before_correction'] = targets_df[targets_df['target'] == 1]['days_to_correction'].mean()
            
            # Correction magnitude distribution
            magnitude_dist = targets_df[targets_df['target'] == 1]['correction_magnitude'].describe()
            validation['correction_magnitude_stats'] = magnitude_dist.to_dict()
            
            # Category distribution
            category_dist = targets_df[targets_df['target'] == 1]['correction_category'].value_counts()
            validation['correction_category_distribution'] = category_dist.to_dict()
        
        # Check for data leakage (targets too close together)
        target_indices = targets_df[targets_df['target'] == 1].index
        if len(target_indices) > 1:
            gaps = target_indices[1:].values - target_indices[:-1].values
            validation['min_gap_between_targets'] = int(gaps.min())
            validation['avg_gap_between_targets'] = float(gaps.mean())
            
            # Flag potential leakage (gaps < 3 days)
            leakage_count = (gaps < 3).sum()
            validation['potential_leakage_count'] = int(leakage_count)
            validation['has_potential_leakage'] = leakage_count > 0
        
        # Quality checks
        validation['quality_checks'] = {
            'has_sufficient_targets': target_days >= 10,
            'balanced_ratio': 0.01 <= validation['target_ratio'] <= 0.3,
            'no_data_leakage': not validation.get('has_potential_leakage', False),
            'reasonable_timing': validation.get('avg_days_before_correction', 0) >= 1.0
        }
        
        print(f"✅ Validation complete:")
        print(f"   📊 Target ratio: {validation['target_ratio']:.3f} ({target_days}/{total_days})")
        print(f"   ⏰ Avg days before correction: {validation.get('avg_days_before_correction', 0):.1f}")
        print(f"   🔍 Quality checks passed: {sum(validation['quality_checks'].values())}/{len(validation['quality_checks'])}")
        
        return validation
    
    def export_targets(self, targets_df: pd.DataFrame, output_path: str):
        """Export targets to file for model training"""
        try:
            # Ensure output directory exists
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Set date as index for better time series handling
            export_df = targets_df.set_index('date')
            
            # Export as parquet (preferred for ML)
            if output_path.suffix == '.parquet' or output_path.suffix == '':
                parquet_path = output_path.with_suffix('.parquet')
                export_df.to_parquet(parquet_path)
                print(f"💾 Exported targets to {parquet_path}")
                
                # Also export as CSV for easy inspection
                csv_path = output_path.with_suffix('.csv')
                export_df.to_csv(csv_path)
                print(f"📄 Also saved as CSV: {csv_path}")
                
            else:
                export_df.to_csv(output_path)
                print(f"💾 Exported targets to {output_path}")
            
            # Export metadata
            metadata_path = output_path.with_suffix('.json')
            metadata = {
                'export_date': datetime.now().isoformat(),
                'total_records': len(export_df),
                'target_count': int(export_df['target'].sum()),
                'date_range': {
                    'start': str(export_df.index.min()),
                    'end': str(export_df.index.max())
                },
                'columns': list(export_df.columns),
                'target_distribution': export_df['target'].value_counts().to_dict()
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"📋 Exported metadata to {metadata_path}")
            
        except Exception as e:
            print(f"❌ Error exporting targets: {e}")
            raise