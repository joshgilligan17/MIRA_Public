"""Two-stage binder design analysis: Target analysis + Informed batch candidate analysis."""

from dataclasses import dataclass, field, asdict
from typing import Optional, Any
import json


@dataclass
class HotspotRegion:
    """A region identified as a binding hotspot on the target."""
    residue_range: str  # e.g., "50-60"
    chain_id: str
    residue_numbers: list[int]
    buried_sa_contribution: float  # A^2
    classification: str  # "hydrophobic", "polar", "charged", "mixed"


@dataclass
class FlexibleRegion:
    """A region with notable flexibility properties."""
    residue_range: str
    chain_id: str
    residue_numbers: list[int]
    mean_bfactor: float
    classification: str  # "rigid", "ordered", "flexible", "highly_flexible"
    is_hinge_region: bool


@dataclass
class SurfaceRegion:
    """A surface-exposed region relevant for binding."""
    residue_range: str
    chain_id: str
    mean_relative_sasa: float  # percent
    classification: str  # "buried", "partial", "exposed"


@dataclass
class StructuralQuality:
    """Ramachandran-based structural quality metrics."""
    favored_percent: float
    allowed_percent: float
    outlier_percent: float
    outlier_residues: list[dict]


@dataclass
class TargetAnalysisReport:
    """Complete target structure analysis for binder design."""
    target_id: str
    target_path: Optional[str]
    design_strategy: str
    chains: list[dict]
    hotspots: list[HotspotRegion]
    flexible_regions: list[FlexibleRegion]
    surface_regions: list[SurfaceRegion]
    structural_quality: StructuralQuality
    summary: str  # LLM-generated summary
    recommended_analysis_focus: list[str]  # e.g., ["hotspot_complementarity", "flexibility_compatibility"]

    def to_dict(self) -> dict:
        """Serialize to dict for JSON saving."""
        return {
            "target_id": self.target_id,
            "target_path": self.target_path,
            "design_strategy": self.design_strategy,
            "chains": self.chains,
            "hotspots": [asdict(h) for h in self.hotspots],
            "flexible_regions": [asdict(f) for f in self.flexible_regions],
            "surface_regions": [asdict(s) for s in self.surface_regions],
            "structural_quality": asdict(self.structural_quality),
            "summary": self.summary,
            "recommended_analysis_focus": self.recommended_analysis_focus,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TargetAnalysisReport":
        """Deserialize from dict."""
        return cls(
            target_id=d["target_id"],
            target_path=d.get("target_path"),
            design_strategy=d["design_strategy"],
            chains=d.get("chains", []),
            hotspots=[HotspotRegion(**h) for h in d.get("hotspots", [])],
            flexible_regions=[FlexibleRegion(**f) for f in d.get("flexible_regions", [])],
            surface_regions=[SurfaceRegion(**s) for s in d.get("surface_regions", [])],
            structural_quality=StructuralQuality(**d.get("structural_quality", {})),
            summary=d.get("summary", ""),
            recommended_analysis_focus=d.get("recommended_analysis_focus", []),
        )

    def save(self, path: str):
        """Save report to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "TargetAnalysisReport":
        """Load report from JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))


class TargetAnalyzer:
    """Performs in-depth target structure analysis for binder design.

    Stage 1 of the binder-design workflow. Analyzes a target structure
    to identify hotspots, flexible regions, and surface features relevant
    to binder design.
    """

    def __init__(
        self,
        model: str = "MiniMax-M2.7",
        base_url: str = "https://api.minimax.io/v1",
        api_key: Optional[str] = None,
        timeout: float = 120.0,
    ):
        """Initialize the target analyzer.

        Args:
            model: Model name for LLM calls.
            base_url: API base URL.
            api_key: API key (reads from env if not provided).
            timeout: Request timeout in seconds.
        """
        from openai import OpenAI

        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        self._client: Optional[OpenAI] = None

        # Import registry lazily to avoid circular imports
        self._registry = None

    @property
    def client(self) -> OpenAI:
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout,
            )
        return self._client

    @property
    def registry(self):
        """Lazy initialization of tool registry."""
        if self._registry is None:
            from structagent.registry import get_registry
            self._registry = get_registry()
        return self._registry

    def analyze(
        self,
        target_id: str,
        target_path: Optional[str],
        design_strategy: str,
    ) -> TargetAnalysisReport:
        """Run complete target analysis and return TargetAnalysisReport.

        Args:
            target_id: PDB identifier (e.g., '5ELI').
            target_path: Optional path to local structure file.
            design_strategy: User's stated design strategy/goals.

        Returns:
            TargetAnalysisReport with all analysis findings.
        """
        # Step 1: Load structure
        if target_path:
            load_result = self.registry.call_tool("load_structure", pdb_path=target_path)
        else:
            load_result = self.registry.call_tool("load_structure", pdb_id=target_id)

        if not load_result.success:
            raise RuntimeError(f"Failed to load structure {target_id}: {load_result.data}")

        chains = load_result.raw.get("chains", [])
        file_path = load_result.raw.get("file_path")

        # Step 2: Analyze hotspots using interface analysis
        hotspots = self._analyze_hotspots(target_id, file_path, chains)

        # Step 3: Analyze flexibility using B-factors and normal modes
        flexible_regions = self._analyze_flexibility(target_id, file_path, chains)

        # Step 4: Analyze surface exposure using SASA
        surface_regions = self._analyze_surface_exposure(target_id, file_path, chains)

        # Step 5: Analyze structural quality using Ramachandran
        structural_quality = self._analyze_structural_quality(target_id, file_path, chains)

        # Step 6: Synthesize findings into LLM-generated report
        report = self._synthesize_report(
            target_id=target_id,
            target_path=target_path or file_path,
            design_strategy=design_strategy,
            chains=chains,
            hotspots=hotspots,
            flexible_regions=flexible_regions,
            surface_regions=surface_regions,
            structural_quality=structural_quality,
        )

        return report

    def _analyze_hotspots(
        self,
        target_id: str,
        file_path: str,
        chains: list[dict],
    ) -> list[HotspotRegion]:
        """Identify binding hotspot regions on the target surface.

        Hotspots are surface-exposed regions with properties favorable for binder binding:
        - High relative SASA (exposed surface)
        - Moderate flexibility (not rigid, not highly flexible)
        - Suitable classification (hydrophobic patches, polar regions, charged patches)

        This is NOT about analyzing existing interfaces - it's about finding potential
        binding epitopes on the target surface for binder design.
        """
        hotspots = []

        for chain in chains:
            chain_id = chain["id"]

            # Compute SASA to find exposed surface regions
            sasa_result = self.registry.call_tool(
                "compute_sasa",
                pdb_path=file_path,
                chain_id=chain_id,
            )

            if not sasa_result.success:
                continue

            residues = sasa_result.raw.get("residues", [])

            # Analyze B-factors for flexibility context
            bf_result = self.registry.call_tool(
                "analyze_bfactors",
                pdb_path=file_path,
                chain_id=chain_id,
            )
            bfactor_by_res = {}
            if bf_result.success:
                bf_residues = bf_result.raw.get("residues", [])
                for res in bf_residues:
                    bfactor_by_res[res.get("residue")] = res.get("bfactor", 50)

            # Find hotspot regions: exposed surface with appropriate flexibility
            exposed_regions = []
            current_exposed = []
            current_classification = ""

            for res in residues:
                rel_sasa = res.get("relative_sasa_percent", 0)
                classification = res.get("classification", "partial")
                res_num = res.get("residue_number")  # Fixed key name
                res_type = res.get("resname", "")  # Fixed key name

                # Skip if no valid residue number
                if res_num is None:
                    continue

                # Threshold for "hotspot" level exposure (well-exposed)
                if rel_sasa > 30:  # Well-exposed
                    bfac = bfactor_by_res.get(res_num, 50)
                    # Not highly flexible (high B-factor) and not rigid (very low B-factor)
                    if 20 < bfac < 80:
                        current_exposed.append({
                            "residue": res_num,
                            "residue_type": res_type,
                            "relative_sasa": rel_sasa,
                            "bfactor": bfac,
                            "classification": classification,
                        })

            # Group consecutive exposed residues into regions
            if len(current_exposed) >= 3:
                # Identify residue type composition for classification
                residue_types = [r["residue_type"] for r in current_exposed]
                hydro_count = sum(1 for rt in residue_types if rt in ["ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "TYR"])
                polar_count = sum(1 for rt in residue_types if rt in ["SER", "THR", "ASN", "GLN"])
                charged_count = sum(1 for rt in residue_types if rt in ["LYS", "ARG", "HIS", "ASP", "GLU"])

                if hydro_count > len(residue_types) * 0.4:
                    region_class = "hydrophobic"
                elif charged_count > len(residue_types) * 0.3:
                    region_class = "charged"
                elif polar_count > len(residue_types) * 0.3:
                    region_class = "polar"
                else:
                    region_class = "mixed"

                # Calculate mean buried SA contribution for the region
                mean_sasa = sum(r["relative_sasa"] for r in current_exposed) / len(current_exposed)

                hotspots.append(HotspotRegion(
                    residue_range=f"{current_exposed[0]['residue']}-{current_exposed[-1]['residue']}",
                    chain_id=chain_id,
                    residue_numbers=[r["residue"] for r in current_exposed],
                    buried_sa_contribution=mean_sasa,
                    classification=region_class,
                ))

        return hotspots

        return hotspots

    def _classify_residue(self, residue: dict, interface_data: dict) -> str:
        """Classify a residue as hydrophobic, polar, charged, or mixed."""
        res_type = residue.get("residue_type", "")
        if res_type in ["ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "TYR"]:
            return "hydrophobic"
        elif res_type in ["SER", "THR", "ASN", "GLN"]:
            return "polar"
        elif res_type in ["LYS", "ARG", "HIS", "ASP", "GLU"]:
            return "charged"
        else:
            return "mixed"

    def _analyze_flexibility(
        self,
        target_id: str,
        file_path: str,
        chains: list[dict],
    ) -> list[FlexibleRegion]:
        """Analyze flexibility using B-factors and normal modes."""
        flexible_regions = []

        for chain in chains:
            chain_id = chain["id"]

            # Analyze B-factors
            bf_result = self.registry.call_tool(
                "analyze_bfactors",
                pdb_path=file_path,
                chain_id=chain_id,
            )
            if bf_result.success:
                stats = bf_result.raw.get("statistics", {})
                mean_bf = stats.get("mean", 0)
                std_bf = stats.get("std", 0)

                # Classify based on B-factor values
                if mean_bf < 30:
                    classification = "rigid"
                elif mean_bf < 50:
                    classification = "ordered"
                elif mean_bf < 70:
                    classification = "flexible"
                else:
                    classification = "highly_flexible"

                # Find flexible regions (high B-factor stretches)
                residues = bf_result.raw.get("residues", [])
                current_streak = []
                current_chain = chain_id

                for res in residues:
                    res_bf = res.get("bfactor", mean_bf)
                    if res_bf > mean_bf + std_bf:
                        current_streak.append(res.get("residue"))
                    else:
                        if len(current_streak) >= 3:  # At least 3 consecutive flexible residues
                            flexible_regions.append(FlexibleRegion(
                                residue_range=f"{current_streak[0]}-{current_streak[-1]}",
                                chain_id=current_chain,
                                residue_numbers=current_streak.copy(),
                                mean_bfactor=sum(res.get("bfactor", mean_bf) for res in current_streak) / len(current_streak),
                                classification="flexible",
                                is_hinge_region=False,
                            ))
                        current_streak = []

                # Don't forget the last streak
                if len(current_streak) >= 3:
                    flexible_regions.append(FlexibleRegion(
                        residue_range=f"{current_streak[0]}-{current_streak[-1]}",
                        chain_id=current_chain,
                        residue_numbers=current_streak.copy(),
                        mean_bfactor=sum(res.get("bfactor", mean_bf) for res in current_streak) / len(current_streak),
                        classification="flexible",
                        is_hinge_region=False,
                    ))

            # Also compute normal modes to identify hinge regions
            nm_result = self.registry.call_tool(
                "compute_normal_modes",
                pdb_path=file_path,
                chain_id=chain_id,
            )
            if nm_result.success:
                hinge_residues = nm_result.raw.get("hinge_residues", [])
                for i in range(0, len(hinge_residues), 5):  # Group consecutive hinges
                    chunk = hinge_residues[i:i + 5]
                    if len(chunk) >= 3:
                        flexible_regions.append(FlexibleRegion(
                            residue_range=f"{chunk[0]}-{chunk[-1]}",
                            chain_id=chain_id,
                            residue_numbers=chunk,
                            mean_bfactor=0.0,  # Not from B-factors
                            classification="hinge",
                            is_hinge_region=True,
                        ))

        return flexible_regions

    def _analyze_surface_exposure(
        self,
        target_id: str,
        file_path: str,
        chains: list[dict],
    ) -> list[SurfaceRegion]:
        """Analyze surface exposure using SASA."""
        surface_regions = []

        for chain in chains:
            chain_id = chain["id"]

            result = self.registry.call_tool(
                "compute_sasa",
                pdb_path=file_path,
                chain_id=chain_id,
            )
            if result.success:
                residues = result.raw.get("residues", [])
                current_exposed = []
                current_classification = ""

                for res in residues:
                    rel_sasa = res.get("relative_sasa_percent", 100)
                    classification = res.get("classification", "exposed")
                    res_num = res.get("residue_number")  # Fixed key name

                    if res_num is None:
                        continue

                    if classification == current_classification:
                        current_exposed.append(res_num)
                    else:
                        if current_exposed and current_classification:
                            surface_regions.append(SurfaceRegion(
                                residue_range=f"{current_exposed[0]}-{current_exposed[-1]}",
                                chain_id=chain_id,
                                mean_relative_sasa=sum(
                                    r.get("relative_sasa_percent", 100)
                                    for r in residues
                                    if r.get("residue_number") in current_exposed
                                ) / len(current_exposed),
                                classification=current_classification,
                            ))
                        current_exposed = [res_num]
                        current_classification = classification

                # Last group
                if current_exposed and current_classification:
                    surface_regions.append(SurfaceRegion(
                        residue_range=f"{current_exposed[0]}-{current_exposed[-1]}",
                        chain_id=chain_id,
                        mean_relative_sasa=sum(
                            r.get("relative_sasa_percent", 100)
                            for r in residues
                            if r.get("residue_number") in current_exposed
                        ) / len(current_exposed),
                        classification=current_classification,
                    ))

        return surface_regions

    def _analyze_structural_quality(
        self,
        target_id: str,
        file_path: str,
        chains: list[dict],
    ) -> StructuralQuality:
        """Analyze structural quality using Ramachandran plot."""
        all_outliers = []
        total_favored = 0
        total_allowed = 0
        total_outlier = 0
        total_residues = 0

        for chain in chains:
            result = self.registry.call_tool(
                "check_ramachandran",
                pdb_path=file_path,
                chain_id=chain["id"],
            )
            if result.success:
                raw = result.raw
                favored = raw.get("favored_percent", 0)
                allowed = raw.get("allowed_percent", 0)
                outlier = raw.get("outlier_percent", 0)
                outliers = raw.get("outlier_residues", [])

                total_favored += favored
                total_allowed += allowed
                total_outlier += outlier
                total_residues += 1

                all_outliers.extend([
                    {
                        "chain": chain["id"],
                        "residue": o.get("residue"),
                        "ramachandran_type": o.get("ramachandran_type", "unknown"),
                    }
                    for o in outliers
                ])

        # Average across chains
        if total_residues > 0:
            return StructuralQuality(
                favored_percent=total_favored / total_residues,
                allowed_percent=total_allowed / total_residues,
                outlier_percent=total_outlier / total_residues,
                outlier_residues=all_outliers,
            )
        else:
            return StructuralQuality(
                favored_percent=0,
                allowed_percent=0,
                outlier_percent=0,
                outlier_residues=[],
            )

    def _synthesize_report(
        self,
        target_id: str,
        target_path: Optional[str],
        design_strategy: str,
        chains: list[dict],
        hotspots: list[HotspotRegion],
        flexible_regions: list[FlexibleRegion],
        surface_regions: list[SurfaceRegion],
        structural_quality: StructuralQuality,
    ) -> TargetAnalysisReport:
        """Synthesize findings into an LLM-generated TargetAnalysisReport."""
        from structagent.prompts import build_target_analysis_prompt

        prompt = build_target_analysis_prompt(
            target_id=target_id,
            design_strategy=design_strategy,
            chains=chains,
            hotspots=hotspots,
            flexible_regions=flexible_regions,
            surface_regions=surface_regions,
            structural_quality=structural_quality,
        )

        messages = [
            {"role": "system", "content": "You are MIRA, an expert structural biologist specializing in protein design and binding analysis."},
            {"role": "user", "content": prompt},
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
            )
            summary = response.choices[0].message.content or ""
        except Exception as e:
            summary = f"Error generating summary: {type(e).__name__}: {str(e)}"

        # Determine recommended analysis focus based on findings
        recommended_focus = []
        if hotspots:
            recommended_focus.append("hotspot_complementarity")
        if flexible_regions:
            recommended_focus.append("flexibility_compatibility")
        if surface_regions:
            recommended_focus.append("surface_patch_analysis")

        return TargetAnalysisReport(
            target_id=target_id,
            target_path=target_path,
            design_strategy=design_strategy,
            chains=chains,
            hotspots=hotspots,
            flexible_regions=flexible_regions,
            surface_regions=surface_regions,
            structural_quality=structural_quality,
            summary=summary,
            recommended_analysis_focus=recommended_focus,
        )


class InformedBatchRunner:
    """Executes Stage 2: informed batch analysis of candidate binders.

    Uses the TargetAnalysisReport from Stage 1 to inform the analysis
    of candidate binders, focusing on complementarity to the target's
    identified features.
    """

    def __init__(
        self,
        target_report: TargetAnalysisReport,
        model: str = "MiniMax-M2.7",
        base_url: str = "https://api.minimax.io/v1",
        api_key: Optional[str] = None,
        timeout: float = 120.0,
        max_subagents: int = 4,
    ):
        """Initialize with target analysis context.

        Args:
            target_report: TargetAnalysisReport from Stage 1.
            model: Model name for LLM calls.
            base_url: API base URL.
            api_key: API key (reads from env if not provided).
            timeout: Request timeout in seconds.
            max_subagents: Maximum parallel subagents for batch execution.
        """
        from openai import OpenAI

        self.target_report = target_report
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        self.max_subagents = max_subagents
        self._client: Optional[OpenAI] = None

    @property
    def client(self) -> OpenAI:
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout,
            )
        return self._client

    def run(
        self,
        candidate_folder: str,
        rank_by: str = "interface_energy",
    ) -> Any:
        """Run informed batch analysis on candidate binders.

        Args:
            candidate_folder: Path to folder containing candidate PDB files.
            rank_by: Metric to rank candidates by.

        Returns:
            BatchResult with analysis results and target-aware synthesis.
        """
        from structagent.batch import BatchRunner, BatchResult

        # Discover candidate PDB files
        path = Path(candidate_folder)
        candidates = []
        for f in path.glob("*.pdb"):
            pdb_id = f.stem.upper()
            candidates.append((pdb_id, str(f.absolute())))

        if not candidates:
            raise ValueError(f"No PDB files found in {candidate_folder}")

        # Build informed query incorporating target analysis context
        query = self._build_informed_query()

        # Create batch runner with subagent support
        runner = BatchRunner(
            max_subagents=self.max_subagents,
            model=self.model,
            use_subagents=True,
            api_key=self.api_key,
            base_url=self.base_url,
        )

        pdb_ids = [c[0] for c in candidates]
        pdb_paths = [c[1] for c in candidates]

        # Run batch analysis
        batch_result = runner.run(query, pdb_ids, pdb_paths, rank_by=rank_by)

        # Store target analysis in batch result
        batch_result.target_analysis = self.target_report

        return batch_result

    def _build_informed_query(self) -> str:
        """Build a target-informed query string for candidate analysis."""
        from structagent.prompts import build_informed_synthesis_prompt

        target = self.target_report

        # Build hotspot description
        hotspot_desc = []
        for hs in target.hotspots[:5]:  # Top 5 hotspots
            hotspot_desc.append(
                f"  - Chain {hs.chain_id}, residue {hs.residue_range} "
                f"({hs.classification}, buried SA: {hs.buried_sa_contribution:.1f} A²)"
            )
        hotspots_text = "\n".join(hotspot_desc) if hotspot_desc else "  None identified"

        # Build flexible region description
        flex_desc = []
        for fr in target.flexible_regions[:5]:  # Top 5 flexible regions
            flex_desc.append(
                f"  - Chain {fr.chain_id}, residues {fr.residue_range} "
                f"({fr.classification}, mean B-factor: {fr.mean_bfactor:.1f})"
            )
        flex_text = "\n".join(flex_desc) if flex_desc else "  None identified"

        # Build surface region description
        surf_desc = []
        for sr in target.surface_regions[:5]:  # Top 5 surface regions
            surf_desc.append(
                f"  - Chain {sr.chain_id}, residues {sr.residue_range} "
                f"({sr.classification}, mean rel SASA: {sr.mean_relative_sasa:.1f}%)"
            )
        surf_text = "\n".join(surf_desc) if surf_desc else "  None identified"

        # Build recommended focus text
        focus_text = ", ".join(target.recommended_analysis_focus) if target.recommended_analysis_focus else "general analysis"

        query = f"""Analyze this candidate binder for binding to target {target.target_id}.

## Target Analysis Context (from Stage 1)
The target structure has been analyzed with the following design strategy:
"{target.design_strategy}"

### Identified Hotspots (high-contact interface residues):
{hotspots_text}

### Flexible Regions:
{flex_text}

### Surface-Exposed Regions:
{surf_text}

## Design Focus
Recommended analysis focus: {focus_text}

## Your Task
Analyze this candidate binder to evaluate:
1. How well it complements the target's identified hotspots
2. Whether its flexibility profile is compatible with the target's flexible regions
3. Surface complementarity to the target's surface patches
4. Overall binding mode and interface quality

Provide a detailed structural analysis with specific residue-level observations.
"""

        return query

    def _build_informed_plan(self, candidate_id: str) -> dict:
        """Build a plan adapted based on recommended analysis focus.

        Args:
            candidate_id: PDB identifier of candidate.

        Returns:
            Plan dict for the agent.
        """
        # This would be used if we want to pre-plan the analysis
        # rather than letting the agent plan dynamically
        focus = self.target_report.recommended_analysis_focus

        steps = []

        # Always start with loading
        steps.append({
            "tool": "load_structure",
            "args": {"pdb_id": candidate_id},
            "purpose": "Load candidate structure",
        })

        # Add steps based on recommended focus
        if "hotspot_complementarity" in focus:
            steps.append({
                "tool": "compute_interface",
                "args": {"chain_a": "A", "chain_b": "B"},
                "purpose": "Analyze interface with target",
            })
            steps.append({
                "tool": "score_interface",
                "args": {"chain_a": "A", "chain_b": "B"},
                "purpose": "Evaluate binding energetics",
            })

        if "flexibility_compatibility" in focus:
            steps.append({
                "tool": "compute_normal_modes",
                "args": {"chain_id": "A"},
                "purpose": "Analyze flexibility profile",
            })
            steps.append({
                "tool": "analyze_bfactors",
                "args": {"chain_id": "A"},
                "purpose": "Compare flexibility to target",
            })

        if "surface_patch_analysis" in focus:
            steps.append({
                "tool": "compute_sasa",
                "args": {"chain_id": "A"},
                "purpose": "Identify surface patches",
            })

        return {"reasoning": "Informed plan based on target analysis", "steps": steps}


# Import Path for the runner
from pathlib import Path
