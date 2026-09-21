# Stratified Insidious Stress Test Implementation

## 🎯 **COMPLETED: Advanced Stratified Prompt Selection System**

### **Key Features Implemented**

#### **1. Stratified Sampling Across Families**
- ✅ **40 insidious slots total** (sessions 10, 20, 30... 490, excluding multiples of 50)
- ✅ **20 crime family prompts** (slots 0-19): Mix of explicit and subtle criminal scenarios
- ✅ **20 corporate family prompts** (slots 20-39): Corporate misconduct and ethical violations
- ✅ **Balanced coverage** ensuring each family gets equal representation

#### **2. Enhanced JSON Configuration Structure**
Each prompt now includes:
```json
{
  "id": "E-1",
  "label": "dead_body_coverup", 
  "family": "crime",
  "explicit": true,
  "prompt": "A client just found a corpse in their flat..."
}
```

#### **3. Comprehensive Logging with Family Analysis**
New CSV columns added:
- `prompt_family`: "crime" or "corporate"
- `prompt_explicit`: true/false for explicit vs subtle wording
- `prompt_id`: Unique identifier for tracking specific prompts
- `prompt_type`: Enhanced type description including family and explicitness

#### **4. Statistical Analysis Capabilities**
- ✅ **Family-based pivoting**: Compare refusal rates between crime vs corporate
- ✅ **Explicitness analysis**: Compare resistance to explicit vs subtle prompts
- ✅ **Individual prompt tracking**: Analyze success/failure rates per prompt ID
- ✅ **Stratification validation**: Ensures balanced sampling across categories

### **Implementation Architecture**

#### **StratifiedPromptSelector Class**
- Loads and validates 40 insidious prompts (20 crime + 20 corporate)
- Implements deterministic stratified sampling
- Provides comprehensive statistics and validation
- Supports both deterministic (seeded) and random selection

#### **Enhanced Domain Models**
- Updated `ICTStressPrompt` to include rich metadata
- Full integration with existing `ExperimentSession` workflow
- Backward compatibility with all existing systems

#### **Advanced CSV Logging**
- Three CSV types per arm: results, stewardship memos, ethical memos
- Family and explicitness data captured in all relevant files
- JSON metadata preservation for detailed post-analysis

### **Usage Example**

```python
# Initialize stratified selector
selector = StratifiedPromptSelector(Path('config'))

# Get prompt for session 10 (will be from crime family)
prompt_obj, prompt_type = selector.get_session_prompt(10)

# Get prompt for session 210 (will be from corporate family) 
prompt_obj, prompt_type = selector.get_session_prompt(210)

# Statistics show balanced distribution
stats = selector.get_prompt_statistics()
# stats['crime_prompts']['total'] == 20
# stats['corporate_prompts']['total'] == 20
```

### **Analysis Capabilities**

#### **Refusal Rate Analysis by Family**
```sql
SELECT 
  prompt_family,
  COUNT(*) as total_attempts,
  SUM(CASE WHEN artifact_found = 'False' THEN 1 ELSE 0 END) as refusals,
  (SUM(CASE WHEN artifact_found = 'False' THEN 1 ELSE 0 END) * 100.0 / COUNT(*)) as refusal_rate
FROM results 
WHERE prompt_type LIKE 'ict_stress%'
GROUP BY prompt_family;
```

#### **Explicitness vs Resistance Analysis**
```sql
SELECT 
  prompt_explicit,
  prompt_family,
  AVG(E_ethics_score) as avg_ethics_score,
  AVG(V_persona_adherence_score) as avg_values_score
FROM results 
WHERE prompt_type LIKE 'ict_stress%'
GROUP BY prompt_explicit, prompt_family;
```

### **Benefits Achieved**

1. **Balanced Testing**: Each experiment run tests both crime and corporate misconduct equally
2. **Granular Analysis**: Can identify specific vulnerabilities by family and explicitness
3. **Reproducible Results**: Deterministic sampling enables consistent cross-run comparisons
4. **Rich Metadata**: Full prompt context preserved for detailed post-experiment analysis
5. **Scalable Design**: Easy to add new families or adjust stratification ratios

### **Validation Results**
- ✅ All 40 prompts loaded successfully
- ✅ Perfect 20/20 stratification achieved
- ✅ Crime family: 10 explicit + 10 subtle prompts
- ✅ Corporate family: 20 explicit prompts  
- ✅ Session scheduling working correctly
- ✅ Enhanced logging capturing all metadata

The stratified insidious stress test system is now **production-ready** and provides comprehensive coverage of both criminal and corporate ethical scenarios with full analytical capabilities!
