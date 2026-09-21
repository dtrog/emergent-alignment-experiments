#!/usr/bin/env python3
"""
Demonstration of the new stratified insidious stress test selection system.
Shows how prompts are selected across crime and corporate families with proper logging.
"""

import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from utils import StratifiedPromptSelector
from loggers.enhanced_logger import EnhancedLogger
from models.analysis import PersonaAlignment, CASVector
from datetime import datetime


def main():
    """Demonstrate the stratified prompt selection system."""
    print("🎯 Stratified Insidious Stress Test Demonstration")
    print("=" * 60)
    
    # Initialize the selector
    selector = StratifiedPromptSelector(Path('config'))
    
    # Show statistics
    stats = selector.get_prompt_statistics()
    print(f"\n📊 Prompt Pool Statistics:")
    print(f"   Crime Family: {stats['crime_prompts']['total']} prompts")
    print(f"     - Explicit: {stats['crime_prompts']['explicit']}")
    print(f"     - Subtle: {stats['crime_prompts']['subtle']}")
    print(f"   Corporate Family: {stats['corporate_prompts']['total']} prompts")
    print(f"     - Explicit: {stats['corporate_prompts']['explicit']}")
    print(f"     - Subtle: {stats['corporate_prompts']['subtle']}")
    
    # Validate configuration
    issues = selector.validate_prompt_pools()
    if issues:
        print(f"\n⚠️  Validation Issues:")
        for issue in issues:
            print(f"     - {issue}")
        return 1
    else:
        print(f"\n✅ All validation checks passed!")
    
    print(f"\n🎲 Sample Prompt Selection (sessions 10, 20, 30, 210, 220, 230):")
    print("-" * 60)
    
    # Test sample sessions to show stratification
    test_sessions = [10, 20, 30, 210, 220, 230, 40, 50, 100]  # Mix of insidious, general, and implicit
    
    for session_num in test_sessions:
        prompt_obj, prompt_type = selector.get_session_prompt(session_num, run_seed=42)
        
        print(f"\nSession {session_num:3d}: {prompt_type}")
        
        if hasattr(prompt_obj, 'metadata') and prompt_obj.metadata:
            metadata = prompt_obj.metadata
            family = metadata.get('family', 'N/A')
            explicit = metadata.get('explicit', 'N/A')
            prompt_id = metadata.get('id', 'N/A')
            print(f"             ID: {prompt_id}, Family: {family}, Explicit: {explicit}")
        
        # Show first 80 chars of prompt
        content = prompt_obj.content if hasattr(prompt_obj, 'content') else str(prompt_obj)
        print(f"             Prompt: {content[:80]}...")
    
    print(f"\n🔬 Testing Enhanced Logging with Stratified Data:")
    print("-" * 60)
    
    # Create logger and test logging with stratified prompts
    logger = EnhancedLogger("stratified-test")
    
    # Create a few mock sessions with stratified prompts
    for i, session_num in enumerate([10, 210], 1):
        prompt_obj, prompt_type = selector.get_session_prompt(session_num, run_seed=42)
        
        # Create mock session
        mock_memo = CommitteeStewardshipMemo(
            artifact_created=i == 1,  # First session creates artifact
            artifact_analysis=None,
            ethical_disobedience_analysis=None,
            persona_alignment_analysis=PersonaAlignment(adherence_score=0.8),
            cas_vector=CASVector(
                artifact_created=i == 1,
                t_score=0.7,
                v_score=0.8,
                ethical_abort=i == 2,  # Second session has ethical abort
                e_score=0.6
            ),
            chair_summary=f"Mock analysis for session {session_num}"
        )
        
        session = ExperimentSession(
            session_id=f"test_session_{session_num}",
            arm_name="Mock-Test-Arm",
            session_number=session_num,
            timestamp=datetime.now(),
            original_prompt=prompt_obj,
            subject_response=f"Mock response for session {session_num}",
            committee_memo=mock_memo,
            subject_llm_name="Mock-LLM-v1.0"
        )
        
        logger.log_session(session)
        print(f"   ✓ Logged session {session_num} with {prompt_type}")
    
    logger.close()
    
    # Show what was logged
    results_dir = logger.get_run_directory()
    files = list(results_dir.glob("*.csv"))
    
    print(f"\n📄 Created {len(files)} CSV files:")
    for file in files:
        print(f"   - {file.name}")
        
        # Show header and first line
        try:
            with open(file, 'r') as f:
                lines = f.readlines()
                if len(lines) >= 2:
                    header = lines[0].strip().split(',')
                    data = lines[1].strip().split(',')
                    
                    print(f"     🏷️  Key fields: prompt_type={data[header.index('prompt_type')]}, " +
                          f"family={data[header.index('prompt_family')]}, " +
                          f"explicit={data[header.index('prompt_explicit')]}")
        except Exception as e:
            print(f"     ❌ Error reading: {e}")
    
    print(f"\n🎉 Stratified Prompt Selection System Ready!")
    print(f"📁 Test results: {results_dir}")
    
    return 0


if __name__ == "__main__":
    exit(main())
