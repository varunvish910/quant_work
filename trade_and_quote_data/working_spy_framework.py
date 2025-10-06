#!/usr/bin/env python3
"""
Working SPY Framework for October 6, 2025 Expiry
Creates a complete framework using confirmed available SPY options
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from pathlib import Path
import json
import time
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class WorkingSPYFramework:
    """Complete framework for SPY options analysis"""
    
    def __init__(self, api_key: str = None, output_dir: str = "data/working_spy"):
        self.api_key = api_key or "OWgBGzgOAzjd6Ieuml6iJakY1yA9npku"
        self.base_url = "https://api.polygon.io"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "contracts").mkdir(exist_ok=True)
        (self.output_dir / "analysis").mkdir(exist_ok=True)
        (self.output_dir / "visualizations").mkdir(exist_ok=True)
    
    def get_october_6_spy_options(self) -> pd.DataFrame:
        """Get all SPY options expiring October 6, 2025"""
        
        print(f"📋 Getting SPY options for October 6, 2025...")
        
        url = f"{self.base_url}/v3/reference/options/contracts"
        
        params = {
            'underlying_ticker': 'SPY',
            'expiration_date': '2025-10-06',
            'limit': 1000,  # Get all available contracts
            'apikey': self.api_key
        }
        
        all_contracts = []
        
        try:
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'results' in data and data['results']:
                    all_contracts.extend(data['results'])
                    
                    # Handle pagination if there are more results
                    while 'next_url' in data and data['next_url']:
                        print(f"   Getting more contracts...")
                        next_response = requests.get(data['next_url'], timeout=30)
                        
                        if next_response.status_code == 200:
                            data = next_response.json()
                            if 'results' in data and data['results']:
                                all_contracts.extend(data['results'])
                            else:
                                break
                        else:
                            break
                        
                        time.sleep(0.1)  # Rate limiting
                
                if all_contracts:
                    contracts_df = pd.DataFrame(all_contracts)
                    
                    # Clean and standardize data
                    contracts_df['expiry'] = pd.to_datetime(contracts_df['expiration_date']).dt.date
                    
                    # Save contracts data
                    contracts_file = self.output_dir / "contracts" / "spy_oct6_2025_contracts.parquet"
                    contracts_df.to_parquet(contracts_file, index=False)
                    
                    print(f"✅ Found {len(contracts_df)} SPY options for October 6, 2025")
                    print(f"   Strike range: ${contracts_df['strike_price'].min():.0f} - ${contracts_df['strike_price'].max():.0f}")
                    print(f"   📁 Saved to: {contracts_file}")
                    
                    return contracts_df
                else:
                    print("❌ No contracts found")
                    return pd.DataFrame()
            else:
                print(f"❌ API error: {response.status_code}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"❌ Error fetching contracts: {e}")
            return pd.DataFrame()
    
    def create_synthetic_positioning_data(self, contracts_df: pd.DataFrame) -> pd.DataFrame:
        """Create realistic synthetic positioning data based on real contracts"""
        
        if len(contracts_df) == 0:
            return pd.DataFrame()
        
        print(f"🎲 Creating synthetic positioning data for {len(contracts_df)} contracts...")
        
        # Get current SPY price
        try:
            spy = yf.Ticker('SPY')
            current_price = spy.history(period="1d")["Close"].iloc[-1]
        except:
            current_price = 569.21
        
        print(f"✓ Using SPY price: ${current_price:.2f}")
        
        # Create synthetic trade and positioning data
        positioning_data = []
        
        for _, contract in contracts_df.iterrows():
            strike = contract['strike_price']
            option_type = contract['contract_type']
            ticker = contract['ticker']
            
            # Calculate realistic positioning based on moneyness
            moneyness = strike / current_price
            
            # Simulate dealer positioning patterns
            if option_type == 'call':
                if moneyness < 0.95:  # Deep ITM calls
                    dealer_position = 'short'  # Dealers usually short deep ITM
                    volume_weight = np.random.uniform(50, 200)
                elif moneyness < 1.05:  # ATM calls
                    dealer_position = np.random.choice(['long', 'short'], p=[0.3, 0.7])
                    volume_weight = np.random.uniform(200, 800)
                else:  # OTM calls
                    dealer_position = 'long'  # Dealers often long OTM
                    volume_weight = np.random.uniform(100, 400)
            else:  # puts
                if moneyness > 1.05:  # Deep ITM puts
                    dealer_position = 'short'
                    volume_weight = np.random.uniform(50, 200)
                elif moneyness > 0.95:  # ATM puts
                    dealer_position = np.random.choice(['long', 'short'], p=[0.4, 0.6])
                    volume_weight = np.random.uniform(200, 800)
                else:  # OTM puts
                    dealer_position = 'long'
                    volume_weight = np.random.uniform(100, 400)
            
            # Calculate synthetic Greeks (simplified Black-Scholes approximation)
            time_to_expiry = (datetime(2025, 10, 6).date() - datetime.now().date()).days / 365.0
            
            if time_to_expiry > 0:
                # Simplified option pricing for demo
                intrinsic = max(0, current_price - strike) if option_type == 'call' else max(0, strike - current_price)
                time_value = max(0.01, np.sqrt(time_to_expiry) * 0.2 * current_price * 0.01)
                option_price = intrinsic + time_value
                
                # Simplified Greeks
                if option_type == 'call':
                    delta = 0.5 + (current_price - strike) / (2 * current_price)
                    delta = np.clip(delta, 0.01, 0.99)
                else:
                    delta = -0.5 + (strike - current_price) / (2 * current_price)  
                    delta = np.clip(delta, -0.99, -0.01)
                
                gamma = max(0.001, np.exp(-0.5 * ((current_price - strike) / (0.2 * current_price)) ** 2) / (current_price * 0.2 * np.sqrt(2 * np.pi)))
                vega = current_price * gamma * np.sqrt(time_to_expiry)
                theta = -vega * 0.2 / (2 * np.sqrt(time_to_expiry)) if time_to_expiry > 0 else 0
            else:
                option_price = intrinsic if 'intrinsic' in locals() else 0
                delta = 1 if option_type == 'call' and strike < current_price else 0
                gamma = 0
                vega = 0
                theta = 0
            
            # Apply dealer position multiplier
            position_multiplier = 1 if dealer_position == 'long' else -1
            
            positioning_data.append({
                'ticker': ticker,
                'strike': strike,
                'option_type': option_type,
                'expiry': contract['expiration_date'],
                'underlying': 'SPY',
                'dealer_position': dealer_position,
                'volume_weight': volume_weight,
                'option_price': option_price,
                'dealer_delta': delta * position_multiplier * volume_weight,
                'dealer_gamma': gamma * position_multiplier * volume_weight,
                'dealer_vega': vega * position_multiplier * volume_weight,
                'dealer_theta': theta * position_multiplier * volume_weight,
                'dealer_vanna': gamma * vega * 0.01 * position_multiplier * volume_weight,
                'dealer_charm': -gamma * delta * 0.01 * position_multiplier * volume_weight,
                'spot_price': current_price,
                'moneyness': moneyness,
                'time_to_expiry': time_to_expiry
            })
        
        positioning_df = pd.DataFrame(positioning_data)
        
        # Save positioning data
        positioning_file = self.output_dir / "analysis" / "synthetic_positioning.parquet"
        positioning_df.to_parquet(positioning_file, index=False)
        
        print(f"✅ Created synthetic positioning for {len(positioning_df)} options")
        print(f"📁 Saved to: {positioning_file}")
        
        return positioning_df
    
    def create_interactive_dashboard(self, positioning_df: pd.DataFrame) -> str:
        """Create interactive dashboard for SPY options positioning"""
        
        if len(positioning_df) == 0:
            return ""
        
        print(f"📊 Creating interactive dashboard...")
        
        current_price = positioning_df['spot_price'].iloc[0]
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Dealer Gamma Profile by Strike',
                'Dealer Delta Profile by Strike', 
                'Volume-Weighted Positioning',
                'Greeks Summary by Option Type'
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": True}, {"secondary_y": False}]]
        )
        
        # Gamma Profile
        calls_df = positioning_df[positioning_df['option_type'] == 'call'].sort_values('strike')
        puts_df = positioning_df[positioning_df['option_type'] == 'put'].sort_values('strike')
        
        fig.add_trace(
            go.Scatter(x=calls_df['strike'], y=calls_df['dealer_gamma'],
                      mode='lines+markers', name='Calls Gamma',
                      line=dict(color='green', width=2)), 
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=puts_df['strike'], y=puts_df['dealer_gamma'],
                      mode='lines+markers', name='Puts Gamma',
                      line=dict(color='red', width=2)),
            row=1, col=1
        )
        
        # Delta Profile
        fig.add_trace(
            go.Scatter(x=calls_df['strike'], y=calls_df['dealer_delta'],
                      mode='lines+markers', name='Calls Delta',
                      line=dict(color='blue', width=2)),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(x=puts_df['strike'], y=puts_df['dealer_delta'],
                      mode='lines+markers', name='Puts Delta',
                      line=dict(color='orange', width=2)),
            row=1, col=2
        )
        
        # Volume-weighted positioning
        fig.add_trace(
            go.Bar(x=positioning_df['strike'], y=positioning_df['volume_weight'],
                   name='Volume Weight', marker_color='lightblue'),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=positioning_df['strike'], y=positioning_df['dealer_gamma'],
                      mode='lines', name='Gamma Overlay',
                      line=dict(color='darkgreen', width=3)),
            row=2, col=1, secondary_y=True
        )
        
        # Greeks summary
        greeks_summary = positioning_df.groupby('option_type').agg({
            'dealer_gamma': 'sum',
            'dealer_delta': 'sum',
            'dealer_vega': 'sum',
            'dealer_theta': 'sum'
        }).reset_index()
        
        greeks_melted = greeks_summary.melt(id_vars=['option_type'], var_name='greek', value_name='value')
        
        for option_type in ['call', 'put']:
            data = greeks_melted[greeks_melted['option_type'] == option_type]
            fig.add_trace(
                go.Bar(x=data['greek'], y=data['value'],
                      name=f'{option_type.title()}s',
                      marker_color='green' if option_type == 'call' else 'red'),
                row=2, col=2
            )
        
        # Add SPY price lines
        for row in [1, 2]:
            for col in [1, 2]:
                fig.add_vline(x=current_price, line_dash="dot", line_color="white",
                             annotation_text=f"SPY: ${current_price:.2f}",
                             row=row, col=col)
        
        # Update layout
        fig.update_layout(
            title=f"SPY Options Dealer Positioning - October 6, 2025 Expiry<br>SPY Price: ${current_price:.2f}",
            height=800,
            template='plotly_dark',
            showlegend=True
        )
        
        # Update axes
        fig.update_xaxes(title_text="Strike Price ($)", row=1, col=1)
        fig.update_xaxes(title_text="Strike Price ($)", row=1, col=2)
        fig.update_xaxes(title_text="Strike Price ($)", row=2, col=1)
        fig.update_xaxes(title_text="Greeks", row=2, col=2)
        
        fig.update_yaxes(title_text="Dealer Gamma", row=1, col=1)
        fig.update_yaxes(title_text="Dealer Delta", row=1, col=2)
        fig.update_yaxes(title_text="Volume Weight", row=2, col=1)
        fig.update_yaxes(title_text="Total Exposure", row=2, col=2)
        
        # Save dashboard
        dashboard_file = self.output_dir / "visualizations" / "spy_oct6_2025_dashboard.html"
        fig.write_html(dashboard_file)
        
        print(f"✅ Interactive dashboard created!")
        print(f"📁 Saved to: {dashboard_file}")
        
        return str(dashboard_file)
    
    def generate_summary_report(self, positioning_df: pd.DataFrame) -> str:
        """Generate summary report"""
        
        if len(positioning_df) == 0:
            return ""
        
        current_price = positioning_df['spot_price'].iloc[0]
        
        # Calculate key metrics
        total_gamma = positioning_df['dealer_gamma'].sum()
        total_delta = positioning_df['dealer_delta'].sum()
        total_vega = positioning_df['dealer_vega'].sum()
        total_theta = positioning_df['dealer_theta'].sum()
        
        # Find gamma centroid
        gamma_centroid = (positioning_df['strike'] * positioning_df['dealer_gamma'].abs()).sum() / positioning_df['dealer_gamma'].abs().sum()
        
        # Count strikes
        total_strikes = len(positioning_df)
        call_strikes = len(positioning_df[positioning_df['option_type'] == 'call'])
        put_strikes = len(positioning_df[positioning_df['option_type'] == 'put'])
        
        report = f"""
# SPY Options Positioning Report
## October 6, 2025 Expiry

### Current Market State
- **SPY Price:** ${current_price:.2f}
- **Days to Expiry:** {(datetime(2025, 10, 6).date() - datetime.now().date()).days} days
- **Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

### Options Chain Overview
- **Total Strikes:** {total_strikes:,}
- **Call Options:** {call_strikes:,}
- **Put Options:** {put_strikes:,}
- **Strike Range:** ${positioning_df['strike'].min():.0f} - ${positioning_df['strike'].max():.0f}

### Dealer Positioning Summary
- **Total Dealer Gamma:** {total_gamma:.0f}
- **Total Dealer Delta:** {total_delta:.0f}
- **Total Dealer Vega:** {total_vega:.0f}
- **Total Dealer Theta:** {total_theta:.0f}
- **Gamma Centroid:** ${gamma_centroid:.2f}

### Key Insights
- **Gamma Centroid vs SPY:** {((gamma_centroid - current_price) / current_price * 100):+.1f}%
- **Net Gamma Exposure:** {'Long' if total_gamma > 0 else 'Short'} ({abs(total_gamma):.0f})
- **Net Delta Exposure:** {'Bullish' if total_delta > 0 else 'Bearish'} ({abs(total_delta):.0f})

### Trading Implications
- This is a **synthetic analysis** based on real SPY option contracts
- Shows how dealer positioning might look as we approach October 6, 2025
- Interactive dashboard provides detailed strike-by-strike analysis
- Use this framework to understand options market structure

### Data Sources
- **Options Contracts:** Real SPY options from Polygon API
- **Positioning Data:** Synthetic dealer positioning simulation
- **Greeks:** Simplified Black-Scholes calculations
- **Market Data:** Live SPY pricing from Yahoo Finance
"""
        
        # Save report
        report_file = self.output_dir / "analysis" / "positioning_report.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"📋 Summary report generated")
        print(f"📁 Saved to: {report_file}")
        
        return str(report_file)


def main():
    """Main execution"""
    
    print("🚀 Working SPY Framework for October 6, 2025")
    print("=" * 50)
    
    framework = WorkingSPYFramework()
    
    try:
        # Step 1: Get real SPY options contracts
        contracts_df = framework.get_october_6_spy_options()
        
        if len(contracts_df) == 0:
            print("❌ No contracts found")
            return
        
        # Step 2: Create synthetic positioning data
        positioning_df = framework.create_synthetic_positioning_data(contracts_df)
        
        # Step 3: Create interactive dashboard
        dashboard_file = framework.create_interactive_dashboard(positioning_df)
        
        # Step 4: Generate summary report
        report_file = framework.generate_summary_report(positioning_df)
        
        print(f"\n🎉 Complete SPY Framework Ready!")
        print(f"📊 Contracts: {len(contracts_df):,}")
        print(f"📈 Positioning Data: {len(positioning_df):,} options")
        print(f"📁 Output Directory: {framework.output_dir}")
        
        print(f"\n📋 Files Created:")
        print(f"   • Interactive Dashboard: {dashboard_file}")
        print(f"   • Summary Report: {report_file}")
        print(f"   • Raw Data: {framework.output_dir}/contracts/ and {framework.output_dir}/analysis/")
        
        print(f"\n💡 This framework shows:")
        print(f"   • Real SPY options available for October 6, 2025")
        print(f"   • Synthetic dealer positioning simulation")
        print(f"   • Interactive visualizations for analysis")
        print(f"   • Complete Greeks calculations and market structure")
        
    except Exception as e:
        print(f"❌ Error in framework: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()