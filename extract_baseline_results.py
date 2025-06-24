#!/usr/bin/env python3

import json
import pandas as pd

def extract_baseline_results():
    """Extract performance data from baseline comparison results"""
    
    with open('results/baseline_comparison_20250622_224640.json', 'r') as f:
        data = json.load(f)
    
    domains = []
    results = []
    
    for domain_name, domain_data in data['baseline_comparison_results'].items():
        if 'results' in domain_data:
            domain_results = {
                'Domain': domain_name.replace('_', ' ').title(),
                'Decade': domain_data['results']['decade']['score'],
                'Decade_Consensus': domain_data['results']['decade']['consensus_score'],
                'Decade_Difference': domain_data['results']['decade']['difference_score'],
                '5-Year': domain_data['results']['5year']['score'],
                '5-Year_Consensus': domain_data['results']['5year']['consensus_score'],
                '5-Year_Difference': domain_data['results']['5year']['difference_score'],
                'Gemini Oracle': domain_data['results']['gemini']['score'],
                'Gemini_Consensus': domain_data['results']['gemini']['consensus_score'],
                'Gemini_Difference': domain_data['results']['gemini']['difference_score'],
                'Bayesian-Optimized': domain_data['results']['bayesian_optimization']['score'],
                'Bayesian_Consensus': domain_data['results']['bayesian_optimization']['consensus_score'],
                'Bayesian_Difference': domain_data['results']['bayesian_optimization']['difference_score']
            }
            results.append(domain_results)
            domains.append(domain_name)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Calculate averages for the table
    table_results = []
    
    methods = [
        ('Bayesian-Optimized', 'Bayesian-Optimized', 'Bayesian_Consensus', 'Bayesian_Difference'),
        ('5-Year Baseline', '5-Year', '5-Year_Consensus', '5-Year_Difference'),
        ('Decade Baseline', 'Decade', 'Decade_Consensus', 'Decade_Difference'),
        ('Gemini Oracle', 'Gemini Oracle', 'Gemini_Consensus', 'Gemini_Difference')
    ]
    
    for method_name, score_col, consensus_col, difference_col in methods:
        avg_score = df[score_col].mean()
        avg_consensus = df[consensus_col].mean()
        avg_difference = df[difference_col].mean()
        std_score = df[score_col].std()
        
        table_results.append({
            'Method': method_name,
            'Avg Score': avg_score,
            'Avg Consensus': avg_consensus,
            'Avg Difference': avg_difference,
            'Std Score': std_score
        })
    
    # Create table DataFrame
    table_df = pd.DataFrame(table_results)
    
    print("=== BASELINE COMPARISON RESULTS (0.8*consensus + 0.2*difference) ===")
    print()
    print(table_df.to_string(index=False, float_format='%.3f'))
    print()
    
    # Find best performing method
    best_method = table_df.loc[table_df['Avg Score'].idxmax(), 'Method']
    best_score = table_df['Avg Score'].max()
    print(f"Best performing method: {best_method} (avg: {best_score:.3f})")
    
    return table_df, df, domains

if __name__ == '__main__':
    extract_baseline_results() 