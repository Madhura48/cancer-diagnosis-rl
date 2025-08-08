"""
Architecture Diagram Generator for Dayhoff Framework
Creates professional system architecture diagram for technical report
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np
from pathlib import Path

def create_system_architecture_diagram():
    """Create comprehensive system architecture diagram"""
    
    # Create figure with high resolution
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    fig.suptitle('Dayhoff Framework: Multi-Agent Reinforcement Learning for Cancer Diagnosis\nMadhura Adadande - Prompt Engineering and GenAI', 
                 fontsize=18, fontweight='bold', y=0.95)
    
    # Define colors for different components
    colors = {
        'data': '#E3F2FD',      # Light blue
        'agents': '#E8F5E8',    # Light green  
        'rl': '#FFF3E0',        # Light orange
        'api': '#F3E5F5',       # Light purple
        'output': '#FFEBEE'     # Light red
    }
    
    # Define component positions and sizes
    components = {
        # Data Layer
        'raw_data': (2, 10, 2.5, 0.8, colors['data'], 'Raw Cancer Data\n(Wisconsin Dataset)\n569 Real Patients'),
        'processed_data': (2, 8.5, 2.5, 0.8, colors['data'], 'Processed Genomic Data\n30 Features per Patient\nNormalized & Augmented'),
        
        # Environment Layer
        'environment': (6, 9.2, 3, 1, colors['rl'], 'Cancer Genomics Environment\n85,350 Training Samples\nReward Function Design'),
        
        # Multi-Agent RL Core
        'feature_agent': (1, 6.5, 3, 1.2, colors['agents'], 'Feature Analysis Agent\n(Deep Q-Network)\n183,966 Parameters\n30→512→256→128→30'),
        'diagnosis_agent': (5.5, 6.5, 3, 1.2, colors['agents'], 'Diagnosis Agent\n(Policy Gradients)\n13,730 Parameters\n25→128→64→32→2'),
        'coordination': (3, 4.8, 3, 0.8, colors['agents'], 'Coordination Agent\nMulti-Agent Orchestration\nReward Sharing'),
        
        # Training Components
        'dqn_training': (0.5, 4, 2, 0.8, colors['rl'], 'DQN Training\nExperience Replay\nTarget Networks'),
        'pg_training': (6.5, 4, 2, 0.8, colors['rl'], 'Policy Gradient\nREINFORCE\nBaseline Estimation'),
        
        # External Integration
        'groq_api': (10, 7.5, 2.2, 0.8, colors['api'], 'Groq API\nLive AI Chat\nReal-time Responses'),
        'hf_api': (10, 6.2, 2.2, 0.8, colors['api'], 'Hugging Face API\nBackup AI Chat\nText Generation'),
        
        # Web Interface
        'web_interface': (10, 4.5, 2.2, 1.2, colors['output'], 'Web Interface\nFlask Application\nLive Chat Integration'),
        
        # Results
        'diagnosis_output': (3, 2.5, 3, 0.8, colors['output'], 'Cancer Diagnosis Output\n91.5% Accuracy\nConfidence Scoring'),
        'model_checkpoints': (7, 2.5, 2.5, 0.8, colors['output'], 'Model Checkpoints\nTrained Weights\nPerformance Metrics')
    }
    
    # Draw components
    for name, (x, y, width, height, color, text) in components.items():
        # Create fancy rounded rectangle
        bbox = FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.1",
            facecolor=color,
            edgecolor='black',
            linewidth=1.5,
            alpha=0.9
        )
        ax.add_patch(bbox)
        
        # Add text
        ax.text(x + width/2, y + height/2, text, 
                ha='center', va='center', fontweight='bold', 
                fontsize=9, wrap=True)
    
    # Define connections (arrows)
    connections = [
        # Data flow
        ('raw_data', 'processed_data', 'blue'),
        ('processed_data', 'environment', 'blue'),
        
        # Environment to agents
        ('environment', 'feature_agent', 'green'),
        ('environment', 'diagnosis_agent', 'green'),
        
        # Agent coordination
        ('feature_agent', 'coordination', 'purple'),
        ('diagnosis_agent', 'coordination', 'purple'),
        
        # Training flows
        ('feature_agent', 'dqn_training', 'orange'),
        ('diagnosis_agent', 'pg_training', 'orange'),
        
        # API integrations
        ('coordination', 'groq_api', 'red'),
        ('coordination', 'hf_api', 'red'),
        ('groq_api', 'web_interface', 'red'),
        ('hf_api', 'web_interface', 'red'),
        
        # Outputs
        ('coordination', 'diagnosis_output', 'darkgreen'),
        ('dqn_training', 'model_checkpoints', 'brown'),
        ('pg_training', 'model_checkpoints', 'brown'),
        ('web_interface', 'diagnosis_output', 'purple')
    ]
    
    # Draw arrows
    for start_name, end_name, color in connections:
        if start_name in components and end_name in components:
            start = components[start_name]
            end = components[end_name]
            
            # Calculate connection points
            start_x = start[0] + start[2]/2
            start_y = start[1]
            end_x = end[0] + end[2]/2
            end_y = end[1] + end[3]
            
            # Adjust connection points for better visualization
            if start_x < end_x:  # Arrow going right
                start_x = start[0] + start[2]
                end_x = end[0]
            elif start_x > end_x:  # Arrow going left
                start_x = start[0]
                end_x = end[0] + end[2]
            
            ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                       arrowprops=dict(arrowstyle='->', lw=2, color=color, alpha=0.7))
    
    # Add legend
    legend_elements = [
        patches.Patch(color=colors['data'], label='Data Processing'),
        patches.Patch(color=colors['agents'], label='RL Agents'),
        patches.Patch(color=colors['rl'], label='Training Components'),
        patches.Patch(color=colors['api'], label='External APIs'),
        patches.Patch(color=colors['output'], label='Outputs & Interface')
    ]
    
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))
    
    # Add performance metrics box
    metrics_text = """🎯 PERFORMANCE METRICS
    
📊 Accuracy: 91.5%
🎓 Training: 150,000 episodes
⏱️ Duration: 3-4 hours
🧠 Total Parameters: 197,696
🏥 Dataset: Real cancer patients
🤖 Methods: DQN + Policy Gradients + Multi-Agent"""
    
    ax.text(13, 9, metrics_text, fontsize=10, 
           bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.8),
           verticalalignment='top')
    
    # Add technical details box
    tech_text = """🔧 TECHNICAL STACK
    
• Framework: Dayhoff Multi-Agent RL
• Hardware: NVIDIA RTX 4060 GPU
• Platform: PyTorch + CUDA
• Web: Flask + Live AI APIs
• APIs: Groq + Hugging Face
• Data: Wisconsin Breast Cancer"""
    
    ax.text(13, 6, tech_text, fontsize=10,
           bbox=dict(boxstyle="round,pad=0.5", facecolor='lightcyan', alpha=0.8),
           verticalalignment='top')
    
    # Set axis properties
    ax.set_xlim(0, 16)
    ax.set_ylim(1.5, 11.5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Save the diagram
    results_dir = Path("final_results")
    results_dir.mkdir(exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'system_architecture_diagram.png', dpi=300, bbox_inches='tight')
    plt.savefig(results_dir / 'system_architecture_diagram.pdf', bbox_inches='tight')
    plt.show()
    
    print("✅ Architecture diagram created!")
    print("📁 Saved to: final_results/system_architecture_diagram.png")

def create_simple_flow_diagram():
    """Create simplified flow diagram for quick understanding"""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fig.suptitle('Cancer Diagnosis AI - System Flow\nMadhura Adadande', 
                 fontsize=16, fontweight='bold')
    
    # Simple flow boxes
    boxes = [
        (1, 6, 'Patient\nGenomic Data', '#E3F2FD'),
        (3.5, 6, 'DQN Agent\nFeature Selection', '#E8F5E8'),
        (6, 6, 'Policy Gradient\nDiagnosis', '#FFF3E0'),
        (8.5, 6, 'Cancer Diagnosis\n91.5% Accuracy', '#FFEBEE'),
        (5, 3.5, 'Live AI Chatbot\nGroq/HF APIs', '#F3E5F5'),
        (5, 1, 'Web Interface\nClinical Dashboard', '#E8EAF6')
    ]
    
    # Draw boxes and text
    for x, y, text, color in boxes:
        rect = FancyBboxPatch(
            (x-0.7, y-0.4), 1.4, 0.8,
            boxstyle="round,pad=0.1",
            facecolor=color,
            edgecolor='black',
            linewidth=2
        )
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontweight='bold', fontsize=10)
    
    # Draw flow arrows
    arrows = [
        ((1.7, 6), (2.8, 6)),    # Data to DQN
        ((4.2, 6), (5.3, 6)),    # DQN to Policy Gradient  
        ((6.7, 6), (7.8, 6)),    # Policy Gradient to Output
        ((5, 5.6), (5, 4.1)),    # System to Chatbot
        ((5, 3.1), (5, 1.6))     # Chatbot to Interface
    ]
    
    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=3, color='darkblue'))
    
    # Add metrics
    ax.text(10, 5, '🏆 ACHIEVEMENTS:\n\n• 91.5% Accuracy\n• 150K Episodes\n• 3-4 Hour Training\n• Real Patient Data\n• Live AI Integration', 
           fontsize=11, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'))
    
    ax.set_xlim(0, 11)
    ax.set_ylim(0.5, 7)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('final_results/simple_flow_diagram.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Simple flow diagram created!")

if __name__ == "__main__":
    print("="*60)
    print("🎨 CREATING ARCHITECTURE DIAGRAMS")
    print("="*60)
    print("📐 Generating system architecture diagram...")
    
    create_system_architecture_diagram()
    create_simple_flow_diagram()
    
    print("\n✅ DIAGRAMS COMPLETED!")
    print("📁 Files created:")
    print("   - system_architecture_diagram.png (detailed)")
    print("   - system_architecture_diagram.pdf (print quality)")
    print("   - simple_flow_diagram.png (overview)")
    print("\n🎯 Use these in your technical report and presentation!")