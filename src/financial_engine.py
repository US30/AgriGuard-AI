import pandas as pd
import numpy as np
import os

class FinancialEngine:
    def __init__(self):
        # 1. Load the Historical Yield Data (Government Data)
        data_dir = "data/raw/financial"
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Directory not found: {data_dir}")

        files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        
        # --- FIX: SMART FILE SELECTION ---
        # We look for the file that actually contains yield data, not just the first one.
        yield_file = None
        for f in files:
            if 'yield' in f.lower():
                yield_file = f
                break
        
        # Fallback: If no file has 'yield' in name, try one with 'crop'
        if not yield_file:
            for f in files:
                if 'crop' in f.lower() and 'price' not in f.lower():
                    yield_file = f
                    break
        
        if not yield_file:
            raise FileNotFoundError(f"Could not find a Yield dataset in {data_dir}. Found: {files}")

        file_path = os.path.join(data_dir, yield_file)
        print(f"[Init] Loading Financial Context from: {yield_file}")
        
        self.df = pd.read_csv(file_path)
        
        # Clean column names (strip spaces, remove special chars)
        self.df.columns = [c.strip().replace(' ', '_').replace(':', '') for c in self.df.columns]
        
        # Standardize State/Crop columns if they have weird names
        # Some datasets use "State_Name", others "State"
        for col in self.df.columns:
            if 'state' in col.lower():
                self.df.rename(columns={col: 'State'}, inplace=True)
            if 'crop' in col.lower() and 'year' not in col.lower():
                self.df.rename(columns={col: 'Crop'}, inplace=True)
            if 'yield' in col.lower():
                self.df.rename(columns={col: 'Yield'}, inplace=True)

        # 2. Define Disease Impact Table (The "Risk Logic")
        self.impact_db = {
            'Healthy': 0.0,
            'Early_blight': 0.20,  # 20% Yield Loss
            'Late_blight': 0.40,   # 40% Yield Loss
            'Leaf_mold': 0.15,
            'Septoria_leaf_spot': 0.15,
            'Spider_mites': 0.10,
            'Target_Spot': 0.25,
            'Mosaic_virus': 0.50,  # Severe
            'Yellow_Leaf_Curl_Virus': 0.60,
            'Bacterial_spot': 0.30,
            'Common_rust': 0.25,
            'Northern_Leaf_Blight': 0.35,
            'Black_rot': 0.30,
            'Esca_(Black_Measles)': 0.40,
            'Leaf_scorch': 0.20
        }

        # Market Prices (Estimated INR per Quintal)
        self.market_prices = {
            'Rice': 2200,
            'Maize': 2100,
            'Potato': 1500,
            'Tomato': 3000,
            'Wheat': 2300,
            'Cotton': 6000
        }

    def get_historical_yield(self, state, crop):
        """
        Fetches the average historical yield for a specific State & Crop.
        """
        state = state.title().strip()
        crop = crop.title().strip()
        
        # Filter data (Case insensitive matching)
        # We use strict 'State' column now since we renamed it in __init__
        subset = self.df[
            (self.df['State'].str.contains(state, case=False, na=False)) & 
            (self.df['Crop'].str.contains(crop, case=False, na=False))
        ]
        
        if subset.empty:
            # Fallback 1: Try finding the crop in ANY state (National Avg)
            subset_national = self.df[self.df['Crop'].str.contains(crop, case=False, na=False)]
            if not subset_national.empty:
                return subset_national['Yield'].mean()
            
            # Fallback 2: Hardcoded safe defaults if crop is totally missing from CSV
            defaults = {'Rice': 4.0, 'Wheat': 3.5, 'Maize': 3.0}
            return defaults.get(crop, 5.0)
            
        return subset['Yield'].mean()

    def calculate_risk_profile(self, state, crop, disease_class, land_area_acres):
        """
        The Core Algo: Combines Vision Diagnosis + Financial Data -> Credit Score
        """
        # 1. Get Baseline Metrics
        hectares = land_area_acres * 0.404686
        
        avg_yield_per_ha = self.get_historical_yield(state, crop)
        expected_total_production = avg_yield_per_ha * hectares
        
        market_price = self.market_prices.get(crop, 2000)
        potential_revenue = expected_total_production * market_price

        # 2. Apply Disease Penalty
        disease_name = disease_class.split('___')[-1] if '___' in disease_class else disease_class
        loss_pct = self.impact_db.get(disease_name, 0.10)
        
        if 'healthy' in disease_name.lower():
            loss_pct = 0.0

        adjusted_yield = expected_total_production * (1 - loss_pct)
        actual_revenue = potential_revenue * (1 - loss_pct)
        revenue_at_risk = potential_revenue - actual_revenue

        # 3. Calculate "Agri-Credit Score" (0-100)
        score = 100
        score -= (loss_pct * 100) * 1.5 
        
        if avg_yield_per_ha > 5: 
            score += 5
            
        final_score = max(min(round(100 - (loss_pct * 100 * 1.2)), 100), 20)

        return {
            "State": state,
            "Crop": crop,
            "Disease_Detected": disease_name,
            "Historical_Yield_Ton_Ha": round(avg_yield_per_ha, 2),
            "Yield_Loss_Pct": f"{loss_pct*100:.1f}%",
            "Projected_Revenue_INR": round(actual_revenue, 2),
            "Revenue_at_Risk_INR": round(revenue_at_risk, 2),
            "Credit_Eligibility_Score": final_score,
            "Recommendation": "Approve Loan" if final_score > 70 else "Reject / Require Insurance"
        }

if __name__ == "__main__":
    engine = FinancialEngine()
    
    # Test Case
    test_state = "Maharashtra"
    test_crop = "Maize"
    test_disease = "Corn___Northern_Leaf_Blight" 
    test_land = 5 
    
    print(f"\n--- SIMULATION: {test_state} Farmer ({test_crop}) ---")
    result = engine.calculate_risk_profile(test_state, test_crop, test_disease, test_land)
    
    for k, v in result.items():
        print(f"{k}: {v}")