import {
  CheckCircle2,
  CircleAlert,
  FileText,
  Server,
  SlidersHorizontal,
} from "lucide-react";
import { CSSProperties, ReactNode, useEffect, useRef, useState } from "react";
import {
  API_BASE,
  FocusedRegion,
  RankingRow,
  ResidueFeature,
  StructureResult,
  SynthesisStatus,
} from "./api";

declare global {
  interface Window {
    $3Dmol?: any;
  }
}

export const evidenceKinds = [
  { key: "interface_residues", label: "Interface", color: "#1f6fbf" },
  { key: "hotspots", label: "Hotspots", color: "#d24f45" },
  { key: "buried_residues", label: "Buried", color: "#2f7d55" },
  { key: "exposed_residues", label: "Exposed", color: "#be7a22" },
  { key: "high_bfactor_residues", label: "Flexible", color: "#8b5cf6" },
  { key: "charge_clusters", label: "Charge", color: "#0891b2" },
  { key: "ramachandran_outliers", label: "Geometry", color: "#c026d3" },
];

export function LogoMark({ compact = false }: { compact?: boolean }) {
  return (
    <div className={compact ? "logo-mark compact" : "logo-mark"}>
      <div className="terminal-line">&gt; MIRA --load model:</div>
      <pre aria-hidden="true">{".--..--.\n  \\\\  //\n   \\\\//\n   //\\\\\n  //  \\\\"}</pre>
      <div>
        <div className="brand-word">MIRA</div>
        <div className="brand-subtitle">MOLECULAR INTELLIGENCE & REASONING AGENT</div>
      </div>
    </div>
  );
}

export function StatusPill({ online, synthesis }: { online: boolean | null; synthesis: SynthesisStatus | null }) {
  const apiLabel = online === null ? "checking" : online ? "api online" : "api offline";
  const synthesisLabel =
    online && synthesis?.configured
      ? `${providerLabel(synthesis.provider)} connected`
      : online
        ? "synthesis not configured"
        : "";
  return (
    <span className={`status-pill ${online ? "ok" : online === false ? "bad" : ""}`}>
      <Server size={15} />
      <span>{apiLabel}</span>
      {synthesisLabel && <span className="status-pill-secondary">{synthesisLabel}</span>}
    </span>
  );
}

export function RankingTable({
  ranking,
  structures,
  selectedId,
  onSelect,
}: {
  ranking: RankingRow[];
  structures: StructureResult[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}) {
  if (!structures.length) {
    return <div className="empty-state">No ranked structures yet.</div>;
  }
  const rankById = new Map(ranking.map((row) => [row.pdb_id, row]));
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Rank</th>
            <th>Structure</th>
            <th>Score</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          {structures.map((structure) => {
            const rank = rankById.get(structure.pdb_id);
            return (
              <tr
                key={structure.id}
                className={selectedId === structure.id ? "selected" : ""}
                onClick={() => onSelect(structure.id)}
              >
                <td>{rank?.rank ?? "-"}</td>
                <td>{structure.pdb_id}</td>
                <td>{typeof rank?.score === "number" ? rank.score.toFixed(2) : "-"}</td>
                <td>
                  {structure.success ? (
                    <span className="result-status ok">
                      <CheckCircle2 size={15} /> pass
                    </span>
                  ) : (
                    <span className="result-status bad">
                      <CircleAlert size={15} /> fail
                    </span>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export function MetricsGrid({ structure }: { structure: StructureResult | null }) {
  const metrics = Object.entries(structure?.metrics ?? {}).filter(([key]) => key !== "total_execution_time");
  if (!structure) {
    return <div className="empty-state compact">Select a structure to inspect metrics.</div>;
  }
  if (!metrics.length) {
    return <div className="empty-state compact">No metrics available for this structure.</div>;
  }
  return (
    <div className="metric-grid">
      {metrics.slice(0, 8).map(([key, value]) => (
        <div className="metric-cell" key={key}>
          <span>{key.replace(/_/g, " ")}</span>
          <strong>{typeof value === "number" ? value.toFixed(2) : value}</strong>
        </div>
      ))}
    </div>
  );
}

export function StructureViewer({
  structure,
  activeEvidence,
  focusedRegion,
}: {
  structure: StructureResult | null;
  activeEvidence: string;
  focusedRegion: FocusedRegion | null;
}) {
  const viewerRef = useRef<HTMLDivElement>(null);
  const [viewerState, setViewerState] = useState("waiting");

  useEffect(() => {
    const currentStructure = structure;
    if (!currentStructure || !viewerRef.current) {
      setViewerState("waiting");
      return;
    }
    const renderedStructure: StructureResult = currentStructure;
    let cancelled = false;

    async function renderStructure() {
      if (!window.$3Dmol) {
        setViewerState("3Dmol unavailable");
        return;
      }
      setViewerState("loading");
      const response = await fetch(`${API_BASE}${renderedStructure.structure_url}`, { credentials: "include" });
      if (!response.ok) {
        throw new Error(`viewer fetch ${response.status}`);
      }
      const text = await response.text();
      if (!text.trim()) {
        throw new Error("empty structure file");
      }
      if (cancelled || !viewerRef.current) {
        return;
      }
      viewerRef.current.innerHTML = "";
      const viewer = window.$3Dmol.createViewer(viewerRef.current, { backgroundColor: "#f8fbff" });
      const format = renderedStructure.filename.toLowerCase().endsWith(".cif") ||
        renderedStructure.filename.toLowerCase().endsWith(".mmcif")
        ? "cif"
        : "pdb";
      viewer.addModel(text, format);
      viewer.setStyle({}, { cartoon: { color: "spectrum", opacity: 0.82 } });
      if (format === "pdb" && isBackboneOnlyPdb(text)) {
        viewer.addStyle(
          {},
          {
            stick: { colorscheme: "Jmol", radius: 0.14, opacity: 0.82 },
          },
        );
        viewer.addStyle({ atom: "CA" }, { sphere: { color: "#176db2", radius: 0.34, opacity: 0.9 } });
      }
      for (const residue of residuesForEvidence(renderedStructure, activeEvidence)) {
        const selector = selectorForResidue(residue);
        if (!selector) {
          continue;
        }
        viewer.addStyle(selector, {
          stick: { color: colorForEvidence(activeEvidence), radius: 0.24 },
          sphere: { color: colorForEvidence(activeEvidence), radius: 0.72 },
        });
      }
      const focusedSelector = focusedRegion ? selectorForFocusedRegion(focusedRegion) : null;
      if (focusedSelector) {
        viewer.addStyle(focusedSelector, {
          stick: { color: "#f8c94b", radius: 0.34 },
          sphere: { color: "#f8c94b", radius: 1.05 },
        });
        viewer.zoomTo(focusedSelector);
      } else {
        viewer.zoomTo();
      }
      if (typeof viewer.resize === "function") {
        viewer.resize();
      }
      viewer.render();
      setViewerState("ready");
    }

    renderStructure().catch((error) => {
      setViewerState(error instanceof Error ? error.message : "viewer error");
    });
    return () => {
      cancelled = true;
    };
  }, [
    structure?.id,
    structure?.structure_url,
    activeEvidence,
    focusedRegion?.evidenceKey,
    focusedRegion?.chain,
    focusedRegion?.residueNumber,
  ]);

  return (
    <div className="viewer-frame">
      <div ref={viewerRef} className="viewer-canvas" />
      {viewerState !== "ready" && <div className="viewer-state">{viewerState}</div>}
    </div>
  );
}

function isBackboneOnlyPdb(text: string) {
  const atomNames = new Set<string>();
  let atomCount = 0;
  for (const line of text.split(/\r?\n/)) {
    if (!line.startsWith("ATOM") && !line.startsWith("HETATM")) {
      continue;
    }
    atomCount += 1;
    atomNames.add(line.slice(12, 16).trim().toUpperCase());
    if (atomNames.size > 5 || atomCount > 800) {
      break;
    }
  }
  if (!atomCount) {
    throw new Error("no atoms found");
  }
  return [...atomNames].every((name) => ["N", "CA", "C", "O"].includes(name));
}

export function EvidenceControls({
  structure,
  activeEvidence,
  onChange,
}: {
  structure: StructureResult | null;
  activeEvidence: string;
  onChange: (key: string) => void;
}) {
  return (
    <div className="evidence-panel">
      <div className="evidence-heading">
        <SlidersHorizontal size={17} />
        <span>Evidence</span>
      </div>
      <div className="evidence-grid">
        {evidenceKinds.map((kind) => {
          const count = evidenceCount(structure, kind.key);
          return (
            <button
              key={kind.key}
              type="button"
              className={activeEvidence === kind.key ? "evidence-chip active" : "evidence-chip"}
              style={{ "--chip-color": kind.color } as CSSProperties}
              onClick={() => onChange(kind.key)}
            >
              <span>{kind.label}</span>
              <strong>{count}</strong>
            </button>
          );
        })}
      </div>
    </div>
  );
}

export function ReportPanel({
  markdown,
  onRegionSelect,
}: {
  markdown: string;
  onRegionSelect: (region: FocusedRegion) => void;
}) {
  return (
    <div className="report-panel">
      <div className="report-heading">
        <FileText size={17} />
        <span>Synthesis Report</span>
      </div>
      {markdown ? (
        <RenderedMarkdown markdown={markdown} onRegionSelect={onRegionSelect} className="report-scroll" />
      ) : (
        <div className="empty-state compact">No synthesis report selected.</div>
      )}
    </div>
  );
}

export function RenderedMarkdown({
  markdown,
  onRegionSelect,
  className = "",
}: {
  markdown: string;
  onRegionSelect: (region: FocusedRegion) => void;
  className?: string;
}) {
  const lines = markdown ? markdown.split("\n") : [];
  return (
    <div className={className}>
      {lines.map((line, index) => (
        <ReportLine key={`${line}-${index}`} line={line} onRegionSelect={onRegionSelect} />
      ))}
    </div>
  );
}

function ReportLine({
  line,
  onRegionSelect,
}: {
  line: string;
  onRegionSelect: (region: FocusedRegion) => void;
}) {
  if (!line.trim()) {
    return <div className="report-spacer" />;
  }
  if (line.startsWith("# ")) {
    return <h3 className="report-title">{renderReportInline(line.slice(2), onRegionSelect)}</h3>;
  }
  if (line.startsWith("## ")) {
    return <h4 className="report-section">{renderReportInline(line.slice(3), onRegionSelect)}</h4>;
  }
  if (line.startsWith("### ")) {
    return <h5 className="report-subsection">{renderReportInline(line.slice(4), onRegionSelect)}</h5>;
  }
  if (line.startsWith("#### ")) {
    return <h6 className="report-minihead">{renderReportInline(line.slice(5), onRegionSelect)}</h6>;
  }
  if (line.startsWith("|")) {
    return <pre className="report-table-line">{line}</pre>;
  }
  if (line.startsWith("- ")) {
    return <div className="report-bullet">{renderReportInline(line.slice(2), onRegionSelect)}</div>;
  }
  return <p className="report-paragraph">{renderReportInline(line, onRegionSelect)}</p>;
}

function renderReportInline(text: string, onRegionSelect: (region: FocusedRegion) => void) {
  const pattern = /\[([^\]]+)\]\(mira:\/\/region\/([^/]+)\/([^/]+)\/([^)]+)\)/g;
  const nodes: ReactNode[] = [];
  let lastIndex = 0;
  for (const match of text.matchAll(pattern)) {
    if (match.index > lastIndex) {
      nodes.push(text.slice(lastIndex, match.index));
    }
    const [, label, evidenceKey, chain, residueNumber] = match;
    nodes.push(
      <button
        key={`${evidenceKey}-${chain}-${residueNumber}-${match.index}`}
        type="button"
        className="region-link"
        onClick={() =>
          onRegionSelect({
            evidenceKey: decodeURIComponent(evidenceKey),
            chain: decodeURIComponent(chain),
            residueNumber: decodeURIComponent(residueNumber),
          })
        }
      >
        {label}
      </button>,
    );
    lastIndex = match.index + match[0].length;
  }
  if (lastIndex < text.length) {
    nodes.push(text.slice(lastIndex));
  }
  return nodes.length ? nodes : text;
}

function providerLabel(provider?: string | null) {
  if (!provider) {
    return "LLM";
  }
  if (provider.toLowerCase() === "minimax") {
    return "MiniMax";
  }
  return provider.charAt(0).toUpperCase() + provider.slice(1);
}

function evidenceCount(structure: StructureResult | null, key: string) {
  if (!structure) {
    return 0;
  }
  return structure.features?.[key]?.length ?? 0;
}

function residuesForEvidence(structure: StructureResult, key: string): ResidueFeature[] {
  const values = structure.features?.[key] ?? [];
  if (key === "charge_clusters") {
    return values.flatMap((cluster) => cluster.residues ?? []);
  }
  return values;
}

function selectorForResidue(residue: ResidueFeature): Record<string, string | number> | null {
  if (residue.residue_number === undefined || residue.residue_number === null) {
    return null;
  }
  const selector: Record<string, string | number> = { resi: Number(residue.residue_number) };
  if (residue.chain) {
    selector.chain = residue.chain;
  }
  return selector;
}

function selectorForFocusedRegion(region: FocusedRegion): Record<string, string | number> | null {
  if (region.residueNumber === undefined || region.residueNumber === null) {
    return null;
  }
  const selector: Record<string, string | number> = { resi: Number(region.residueNumber) };
  if (region.chain && region.chain !== "any") {
    selector.chain = region.chain;
  }
  return selector;
}

function colorForEvidence(key: string) {
  return evidenceKinds.find((kind) => kind.key === key)?.color ?? "#1f6fbf";
}
