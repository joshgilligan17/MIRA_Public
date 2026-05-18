"""Enhanced synthesis engine for comparative batch analysis."""

import asyncio
from dataclasses import dataclass
from typing import Optional, Any

from openai import OpenAI

from structagent.batch import StructureResult, ResultAggregator


@dataclass
class SynthesisReport:
    """Complete synthesis report for batch analysis."""

    ranking: list[tuple[str, float]]  # (pdb_id, score) sorted by criterion
    top_constructs: list[dict]  # Top 3-5 constructs with reasons
    comparative_analysis: str  # LLM-generated comparative text
    design_suggestions: list[str]  # Actionable suggestions
    ranking_criterion: str


class BatchSynthesisEngine:
    """Generates comparative synthesis from batch results.

    This engine replaces the basic synthesis in batch.py with a more capable
    version that can:
    1. Compare results across multiple binders
    2. Generate rankings based on configurable criteria
    3. Identify top-k most promising constructs
    4. Suggest favorable design changes
    """

    RANKING_CRITERIA = {
        # SASA-based stability: lower mean relative SASA = more buried/stable
        "stability": {"metric": "mean_relative_sasa_percent", "higher_is_better": False},
        # Interface metrics: higher buried surface area = stronger interaction
        "buried_surface_area": {"metric": "buried_surface_area", "higher_is_better": True},
        # Interface residues: more interface residues = larger interface
        "n_interface_residues": {"metric": "n_interface_residues", "higher_is_better": True},
        # B-factor flexibility: lower mean B-factor = more rigid/stable
        "mean_bfactor": {"metric": "mean_bfactor", "higher_is_better": False},
        # B-factor variability: lower std B-factor = more uniform
        "std_bfactor": {"metric": "std_bfactor", "higher_is_better": False},
        # Buried count: more buried residues = more hydrophobic core
        "n_buried": {"metric": "n_buried", "higher_is_better": True},
        # Exposed count: fewer exposed = more well-packed
        "n_exposed": {"metric": "n_exposed", "higher_is_better": False},
        # Interface energy: lower dG = stronger binding = better
        "interface_energy": {"metric": "interface_energy", "higher_is_better": False},
        # Shape complementarity: higher = better geometric fit
        "shape_complementarity": {"metric": "shape_complementarity", "higher_is_better": True},
        # Packing quality: higher = better van der Waals packing
        "packstat": {"metric": "packstat", "higher_is_better": True},
    }

    CRITERION_DESCRIPTIONS = {
        "stability": "stability (mean relative SASA - lower means more stable/buried residues)",
        "buried_surface_area": "buried surface area (higher means more extensive interface)",
        "n_interface_residues": "interface residue count (higher means larger interface)",
        "mean_bfactor": "flexibility from B-factors (lower means more rigid)",
        "std_bfactor": "B-factor variability (lower means more uniform)",
        "n_buried": "buried residue count (higher means more hydrophobic core)",
        "n_exposed": "exposed residue count (lower means better packing)",
        "interface_energy": "binding energy (lower means stronger binding)",
        "shape_complementarity": "shape complementarity (higher means better geometric fit)",
        "packstat": "packing quality (higher means better van der Waals packing)",
    }

    def __init__(
        self,
        model: str = "MiniMax-M2.7",
        api_key: Optional[str] = None,
        base_url: str = "https://api.minimax.io/v1",
    ):
        """Initialize the synthesis engine.

        Args:
            model: Model identifier for LLM calls
            api_key: API key for the LLM provider
            base_url: Base URL for the API endpoint
        """
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self._client: Optional[OpenAI] = None

    @property
    def client(self) -> OpenAI:
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=120.0,
            )
        return self._client

    async def synthesize(
        self,
        subagent_results: list,
        query: str,
        rank_by: str = "interface_energy",
    ) -> SynthesisReport:
        """Generate synthesis from all subagent results.

        Args:
            subagent_results: List of objects with .structure_results attribute
                              (e.g., BatchResult objects from batch.py)
            query: The original user query that drove the analysis
            rank_by: Criterion for ranking structures

        Returns:
            SynthesisReport with ranking, top constructs, comparative analysis,
            and design suggestions
        """
        # 1. Aggregate all structure results from all subagents
        all_results = []
        for sub_result in subagent_results:
            # Handle both BatchResult (has structure_results) and direct list of StructureResult
            if hasattr(sub_result, "structure_results"):
                all_results.extend(sub_result.structure_results)
            elif isinstance(sub_result, StructureResult):
                all_results.append(sub_result)
            elif isinstance(sub_result, list):
                all_results.extend(sub_result)

        if not all_results:
            return SynthesisReport(
                ranking=[],
                top_constructs=[],
                comparative_analysis="No results to analyze.",
                design_suggestions=[],
                ranking_criterion=rank_by,
            )

        # 2. Rank structures using ResultAggregator pattern
        ranking = self._rank_structures(all_results, rank_by)

        # 3. Identify top constructs with detailed reasons
        top_constructs = self._identify_top_constructs(all_results, ranking, top_k=5)

        # 4. Generate comparative analysis via LLM (async)
        comparative_task = self._generate_comparative_analysis(all_results, ranking, query)

        # 5. Generate design suggestions via LLM (async)
        suggestions_task = self._generate_design_suggestions(all_results, top_constructs, query)

        # Run both LLM calls concurrently
        comparative, suggestions = await asyncio.gather(
            comparative_task,
            suggestions_task,
        )

        return SynthesisReport(
            ranking=ranking,
            top_constructs=top_constructs,
            comparative_analysis=comparative,
            design_suggestions=suggestions,
            ranking_criterion=rank_by,
        )

    def _rank_structures(self, results: list[StructureResult], rank_by: str) -> list[tuple[str, float]]:
        """Rank structures by specified criterion.

        Args:
            results: List of StructureResult objects
            rank_by: The ranking criterion name

        Returns:
            List of (pdb_id, score) tuples sorted by criterion
        """
        aggregator = ResultAggregator(rank_by)
        for result in results:
            aggregator.add_result(result)
        return aggregator.get_ranking()

    def _identify_top_constructs(
        self,
        results: list[StructureResult],
        ranking: list[tuple[str, float]],
        top_k: int = 5,
    ) -> list[dict]:
        """Identify top-k constructs with detailed reasoning.

        Args:
            results: All structure results
            ranking: Sorted ranking list
            top_k: Number of top constructs to identify

        Returns:
            List of dicts with pdb_id, score, metrics, and reason for ranking
        """
        top_constructs = []
        ranking_dict = dict(ranking)

        # Use aggregator to extract metrics for each result
        aggregator = ResultAggregator("interface_energy")  # Dummy criterion for extraction

        for pdb_id, score in ranking[:top_k]:
            # Find the structure result
            result = next((r for r in results if r.pdb_id == pdb_id), None)
            if result is None:
                continue

            # Extract all available metrics
            metrics = aggregator.extract_metrics(result)

            # Build reason for ranking
            reason = self._build_ranking_reason(pdb_id, metrics, score)

            top_constructs.append(
                {
                    "pdb_id": pdb_id,
                    "score": score,
                    "metrics": metrics,
                    "reason": reason,
                    "success": result.success,
                    "error": result.error if not result.success else None,
                }
            )

        return top_constructs

    def _build_ranking_reason(self, pdb_id: str, metrics: dict, score: float) -> str:
        """Build a human-readable reason for why a construct ranks where it does.

        Args:
            pdb_id: Structure identifier
            metrics: Extracted metrics dict
            score: The ranking score

        Returns:
            Human-readable explanation
        """
        reasons = []

        # Interface energy
        if "interface_energy" in metrics:
            dG = metrics["interface_energy"]
            if dG < -20:
                strength = "very strong"
            elif dG < -10:
                strength = "moderate"
            elif dG < 0:
                strength = "weak"
            else:
                strength = "unfavorable"
            reasons.append(f"interface energy of {dG:.1f} REU indicates {strength} binding")

        # Buried surface area
        if "buried_surface_area" in metrics:
            bsa = metrics["buried_surface_area"]
            if bsa > 2000:
                size = "very large"
            elif bsa > 1000:
                size = "medium-large"
            elif bsa > 500:
                size = "moderate"
            else:
                size = "small"
            reasons.append(f"buried surface area of {bsa:.0f} A² indicates {size} interface")

        # Shape complementarity
        if "shape_complementarity" in metrics:
            sc = metrics["shape_complementarity"]
            if sc > 0.7:
                quality = "excellent"
            elif sc > 0.6:
                quality = "good"
            elif sc > 0.5:
                quality = "moderate"
            else:
                quality = "poor"
            reasons.append(f"shape complementarity of {sc:.2f} is {quality}")

        # Stability (SASA)
        if "mean_relative_sasa_percent" in metrics:
            sasa = metrics["mean_relative_sasa_percent"]
            if sasa < 15:
                stability = "highly stable"
            elif sasa < 25:
                stability = "stable"
            elif sasa < 40:
                stability = "moderately stable"
            else:
                stability = "flexible/exposed"
            reasons.append(f"mean relative SASA of {sasa:.1f}% suggests {stability} structure")

        # B-factors
        if "mean_bfactor" in metrics:
            bfac = metrics["mean_bfactor"]
            if bfac < 30:
                rigidity = "very rigid"
            elif bfac < 50:
                rigidity = "rigid"
            elif bfac < 70:
                rigidity = "moderately flexible"
            else:
                rigidity = "highly flexible"
            reasons.append(f"mean B-factor of {bfac:.1f} indicates {rigidity} regions")

        if reasons:
            return f"{pdb_id}: " + "; ".join(reasons)
        else:
            return f"{pdb_id}: score={score:.2f}"

    async def _generate_comparative_analysis(
        self,
        results: list[StructureResult],
        ranking: list[tuple[str, float]],
        query: str,
    ) -> str:
        """Use LLM to generate comparative analysis across all binders.

        Args:
            results: All structure results
            ranking: Sorted ranking list
            query: The original user query

        Returns:
            LLM-generated comparative analysis text
        """
        # Build comprehensive summary of all results
        summary = self._build_results_summary(results, ranking)

        criterion_desc = (
            self.CRITERION_DESCRIPTIONS.get(ranking[0][1] if ranking else "interface_energy", "interface energy")
            if ranking
            else "interface energy"
        )

        prompt = f"""You are MIRA, an expert structural biologist performing comparative synthesis on multiple protein construct designs.

## Original Query
{query}

## Task
You are analyzing {len(results)} binder constructs to determine which are most promising for protein-protein interaction design. Your analysis must be TRULY COMPARATIVE - identify patterns, differences, and relationships across ALL constructs, not just list individual results.

## Results Summary
{summary}

## Ranking Criterion
Structures are ranked by {criterion_desc}.

## Your Synthesis Task

Analyze the results to provide:

1. **COMPARATIVE INSIGHTS**: Compare and contrast the constructs:
   - What patterns emerge across the top performers?
   - What distinguishes the best constructs from the worst?
   - Are there trade-offs between different metrics (e.g., stability vs. binding strength)?

2. **STRUCTURAL INTERPRETATION**: Explain the structural basis for the ranking:
   - What structural features drive favorable metrics?
   - Are there concerning features in poorly-ranked constructs?
   - How do interface properties correlate with overall stability?

3. **KEY DIFFERENTIATORS**: Identify the most important factors that separate:
   - Top 2-3 constructs (why they excel)
   - Bottom 2-3 constructs (what holds them back)
   - Middle constructs (areas for improvement)

4. **BIOLOGICAL IMPLICATIONS**: What do these results suggest about:
   - Likely binding affinity differences
   - Structural stability in different conditions
   - Potential for further optimization

## Output Format

Provide your analysis in structured markdown:

## Comparative Analysis Summary
[2-3 paragraph overview comparing all constructs and explaining the overall ranking pattern]

## Key Comparative Findings
| Finding | Supporting Evidence | Structural Explanation |
|---------|---------------------|------------------------|
| [Pattern 1] | [Metric evidence] | [Structural mechanism] |
| [Pattern 2] | [Metric evidence] | [Structural mechanism] |

## Top Construct Analysis
**Why [Top Construct] ranks #1**: [Detailed structural explanation]
**What makes it successful**: [Specific features and metrics]

## Concerns and Limitations
[Any issues with the analysis, data quality concerns, or limitations in the comparison]
"""

        messages = [
            {
                "role": "system",
                "content": "You are MIRA, an expert structural biologist with deep knowledge of protein structure, function, and design. You provide rigorous, data-driven comparative analysis.",
            },
            {"role": "user", "content": prompt},
        ]

        try:
            response = await asyncio.to_thread(
                self._call_llm,
                messages=messages,
            )
            return response
        except Exception as e:
            return f"Error generating comparative analysis: {type(e).__name__}: {str(e)}"

    async def _generate_design_suggestions(
        self,
        results: list[StructureResult],
        top_constructs: list[dict],
        query: str,
    ) -> list[str]:
        """Use LLM to suggest actionable design improvements.

        Args:
            results: All structure results
            top_constructs: Top-k constructs with reasons
            query: The original user query

        Returns:
            List of actionable design suggestions
        """
        # Build summary of top constructs and overall results
        top_summary = self._build_top_constructs_summary(top_constructs)
        all_summary = self._build_all_metrics_summary(results)

        prompt = f"""You are MIRA, an expert structural biologist providing actionable design suggestions for protein construct optimization.

## Original Design Query
{query}

## Top Performing Constructs (for reference)
{top_summary}

## Overall Metrics Summary Across All Constructs
{all_summary}

## Your Task

Based on the analysis results, provide 5-8 SPECIFIC, ACTIONABLE design suggestions that could improve the weaker constructs or further optimize the stronger ones.

## Guidelines for Good Suggestions

Each suggestion should:
1. **Be specific**: Name actual residues, regions, or structural features when possible
2. **Be actionable**: Describe a concrete change (mutation, deletion, extension) that could be made
3. **Be justified**: Explain WHY this change would help based on the structural data
4. **Consider trade-offs**: Note any potential negative consequences of the suggested change

## Suggestion Categories to Consider

- **Interface optimization**: Hotspot residue mutations, charge complementarity, hydrophobic packing
- **Stability improvements**: Buried hydrophobic residues, salt bridges, disulfide bonds
- **Flexibility adjustments**: Rigidify flexible regions, flexibility in hinge areas
- **Expression/purification**: Reduce surface hydrophobic patches, optimize isoelectric point
- **Specificity enhancements**: Remove ambiguous contacts, improve shape complementarity

## Output Format

Provide suggestions as a numbered list:

1. **[Category] Suggestion for [Target Construct/Region]**
   - **Change**: [Specific modification to make]
   - **Rationale**: [Why this would improve the design based on structural data]
   - **Expected Impact**: [High/Medium/Low and direction]

2. ... etc
"""

        messages = [
            {
                "role": "system",
                "content": "You are MIRA, an expert structural biologist specializing in protein design optimization. You provide specific, actionable suggestions grounded in structural data.",
            },
            {"role": "user", "content": prompt},
        ]

        try:
            response = await asyncio.to_thread(
                self._call_llm,
                messages=messages,
            )
            # Parse suggestions from the response
            return self._parse_suggestions(response)
        except Exception as e:
            return [f"Error generating design suggestions: {type(e).__name__}: {str(e)}"]

    def _call_llm(self, messages: list[dict], temperature: float = 0.0) -> str:
        """Make a synchronous LLM call (will be run in thread pool).

        Args:
            messages: Chat messages
            temperature: Sampling temperature

        Returns:
            LLM response content
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""

    def _build_results_summary(
        self,
        results: list[StructureResult],
        ranking: list[tuple[str, float]],
    ) -> str:
        """Build a text summary of all results for LLM consumption.

        Args:
            results: All structure results
            ranking: Sorted ranking list

        Returns:
            Formatted summary string
        """
        lines = []
        ranking_dict = dict(ranking)

        # Group by ranking position
        for rank, (pdb_id, score) in enumerate(ranking, 1):
            result = next((r for r in results if r.pdb_id == pdb_id), None)
            if result is None:
                continue

            if result.success:
                # Extract metrics
                aggregator = ResultAggregator("interface_energy")
                metrics = aggregator.extract_metrics(result)

                metrics_str = ", ".join(
                    f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in sorted(metrics.items())
                )
                lines.append(f"[Rank {rank}] {pdb_id} (score={score:.2f}): {metrics_str}")
            else:
                lines.append(f"[Rank {rank}] {pdb_id}: FAILED - {result.error}")

        return "\n".join(lines) if lines else "No results available"

    def _build_top_constructs_summary(self, top_constructs: list[dict]) -> str:
        """Build a summary of top constructs.

        Args:
            top_constructs: List from _identify_top_constructs

        Returns:
            Formatted summary string
        """
        if not top_constructs:
            return "No top constructs identified."

        lines = []
        for i, construct in enumerate(top_constructs, 1):
            pdb_id = construct["pdb_id"]
            score = construct["score"]
            reason = construct["reason"]

            metrics = construct.get("metrics", {})
            if metrics:
                metrics_str = ", ".join(
                    f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in sorted(metrics.items())
                )
                lines.append(f"{i}. {pdb_id} (score={score:.2f})")
                lines.append(f"   Metrics: {metrics_str}")
                lines.append(f"   Reason: {reason}")
            else:
                lines.append(f"{i}. {pdb_id} (score={score:.2f}): {reason}")

        return "\n".join(lines)

    def _build_all_metrics_summary(self, results: list[StructureResult]) -> str:
        """Build a summary of all available metrics across results.

        Args:
            results: All structure results

        Returns:
            Formatted summary string
        """
        if not results:
            return "No results available."

        # Collect all metric names that appear
        aggregator = ResultAggregator("interface_energy")
        all_metric_names = set()
        for result in results:
            if result.success:
                metrics = aggregator.extract_metrics(result)
                all_metric_names.update(metrics.keys())

        if not all_metric_names:
            return "No metrics extracted from results."

        # Build summary table
        lines = ["Metric Summary Across All Constructs:"]
        for metric_name in sorted(all_metric_names):
            values = []
            for result in results:
                if result.success:
                    m = aggregator.extract_metrics(result)
                    if metric_name in m:
                        v = m[metric_name]
                        values.append(f"{result.pdb_id}={v:.2f}" if isinstance(v, float) else f"{result.pdb_id}={v}")
            if values:
                lines.append(f"  {metric_name}: {', '.join(values)}")

        return "\n".join(lines)

    def _parse_suggestions(self, response: str) -> list[str]:
        """Parse LLM response into a list of suggestions.

        Args:
            response: Raw LLM response text

        Returns:
            List of suggestion strings
        """
        suggestions = []
        lines = response.split("\n")

        current_suggestion = []
        for line in lines:
            stripped = line.strip()
            # Check if this is a numbered suggestion
            if stripped and stripped[0].isdigit() and ". " in stripped[:4]:
                if current_suggestion:
                    suggestions.append("\n".join(current_suggestion))
                current_suggestion = [stripped]
            elif current_suggestion:
                current_suggestion.append(stripped)

        if current_suggestion:
            suggestions.append("\n".join(current_suggestion))

        # If parsing didn't work well, return the whole response as one item
        if not suggestions or len(suggestions) == 1 and len(response) > 500:
            # Try to extract bullet points
            bullets = [line.strip() for line in lines if line.strip().startswith("-")]
            if bullets:
                return bullets
            return [response]

        return suggestions

    def synthesize_with_target(
        self,
        batch_result: Any,
        provider: Any,
        model: str,
        temperature: float = 0.0,
    ) -> str:
        """Generate target-informed synthesis from batch results.

        Uses the TargetAnalysisReport from Stage 1 to provide context-aware
        comparative analysis of candidate binders.

        Args:
            batch_result: BatchResult with candidate analysis results
            provider: LLM provider for chat completions
            model: Model name for LLM calls
            temperature: Sampling temperature

        Returns:
            LLM-generated target-aware comparative analysis
        """
        from structagent.prompts import build_informed_synthesis_prompt

        target_report = batch_result.target_analysis
        if target_report is None:
            # Fall back to regular synthesis
            return self.synthesize(batch_result, provider, model, temperature)

        prompt = build_informed_synthesis_prompt(
            target_report=target_report,
            batch_result=batch_result,
            ranking=batch_result.ranking,
            ranking_criterion=batch_result.ranking_criterion,
        )

        messages = [
            {"role": "system", "content": "You are MIRA, an expert structural biologist."},
            {"role": "user", "content": prompt},
        ]

        try:
            response = provider.chat(messages=messages, model=model, temperature=temperature)
            return response.content or ""
        except Exception as e:
            return f"Error generating target-informed synthesis: {type(e).__name__}: {str(e)}"
