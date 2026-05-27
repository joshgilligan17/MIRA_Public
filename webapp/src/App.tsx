import {
  Activity,
  ArrowDownToLine,
  CheckCircle2,
  CircleAlert,
  FileArchive,
  FileText,
  Loader2,
  Microscope,
  Play,
  RefreshCw,
  Server,
  SlidersHorizontal,
  Upload,
} from "lucide-react";
import { ChangeEvent, CSSProperties, FormEvent, ReactNode, useEffect, useMemo, useRef, useState } from "react";

const API_BASE = import.meta.env.VITE_MIRA_API_BASE ?? "http://localhost:8000";

declare global {
  interface Window {
    $3Dmol?: any;
  }
}

type Profile = {
  name: string;
  label: string;
  description: string;
  default_rank_by: string;
  tools: string[];
};

type Job = {
  id: string;
  status: "queued" | "running" | "completed" | "failed";
  completed_count: number;
  total_count: number;
  failed_count: number;
  error?: string | null;
  config: {
    query: string;
    profile: string;
    rank_by: string;
  };
};

type RankingRow = {
  rank: number;
  pdb_id: string;
  score: number;
};

type ResidueFeature = {
  kind: string;
  chain?: string | null;
  residue_number?: number | string | null;
  residue_name?: string | null;
  label: string;
  score?: number | null;
};

type StructureResult = {
  id: string;
  pdb_id: string;
  filename: string;
  success: boolean;
  error?: string | null;
  profile: string;
  chains: { id: string; length?: number; first_residue?: number; last_residue?: number }[];
  metrics: Record<string, number | string>;
  features: Record<string, any[]>;
  warnings: string[];
  summary: string;
  structure_url: string;
};

type Results = {
  summary: Record<string, number | string | null>;
  ranking: RankingRow[];
  structures: StructureResult[];
};

type FocusedRegion = {
  evidenceKey: string;
  chain?: string | null;
  residueNumber?: number | string | null;
};

const rankOptions = [
  { value: "stability", label: "Stability" },
  { value: "buried_surface_area", label: "Buried surface" },
  { value: "n_interface_residues", label: "Interface residues" },
  { value: "mean_bfactor", label: "Mean B-factor" },
  { value: "n_buried", label: "Buried count" },
  { value: "n_exposed", label: "Exposed count" },
  { value: "interface_energy", label: "Interface energy" },
];

const providerOptions = [
  { value: "", label: "Auto", model: "", baseUrl: "" },
  { value: "openai", label: "OpenAI", model: "gpt-4o-mini", baseUrl: "https://api.openai.com/v1" },
  { value: "minimax", label: "MiniMax", model: "MiniMax-M2.7", baseUrl: "https://api.minimax.io/v1" },
  { value: "anthropic", label: "Anthropic", model: "claude-3-5-haiku-20241022", baseUrl: "https://api.anthropic.com" },
  { value: "azure", label: "Azure", model: "", baseUrl: "" },
];

const evidenceKinds = [
  { key: "interface_residues", label: "Interface", color: "#1f6fbf" },
  { key: "hotspots", label: "Hotspots", color: "#d24f45" },
  { key: "buried_residues", label: "Buried", color: "#2f7d55" },
  { key: "exposed_residues", label: "Exposed", color: "#be7a22" },
  { key: "high_bfactor_residues", label: "Flexible", color: "#8b5cf6" },
  { key: "charge_clusters", label: "Charge", color: "#0891b2" },
  { key: "ramachandran_outliers", label: "Geometry", color: "#c026d3" },
];

export default function App() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [profiles, setProfiles] = useState<Profile[]>([]);
  const [files, setFiles] = useState<File[]>([]);
  const [query, setQuery] = useState("Rank these structures for de novo filtering.");
  const [profile, setProfile] = useState("triage_default");
  const [rankBy, setRankBy] = useState("stability");
  const [chainA, setChainA] = useState("");
  const [chainB, setChainB] = useState("");
  const [maxWorkers, setMaxWorkers] = useState(2);
  const [enableLlmSynthesis, setEnableLlmSynthesis] = useState(true);
  const [llmProvider, setLlmProvider] = useState("");
  const [llmModel, setLlmModel] = useState("");
  const [llmBaseUrl, setLlmBaseUrl] = useState("");
  const [llmApiKey, setLlmApiKey] = useState("");
  const [jobId, setJobId] = useState<string | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [results, setResults] = useState<Results | null>(null);
  const [reportMarkdown, setReportMarkdown] = useState("");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [activeEvidence, setActiveEvidence] = useState("interface_residues");
  const [focusedRegion, setFocusedRegion] = useState<FocusedRegion | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [apiOnline, setApiOnline] = useState<boolean | null>(null);

  useEffect(() => {
    fetch(`${API_BASE}/api/health`)
      .then((response) => setApiOnline(response.ok))
      .catch(() => setApiOnline(false));

    fetch(`${API_BASE}/api/profiles`)
      .then((response) => response.json())
      .then((data) => setProfiles(data.profiles ?? []))
      .catch(() => setProfiles([]));
  }, []);

  useEffect(() => {
    const id = new URLSearchParams(window.location.search).get("job");
    if (id) {
      setJobId(id);
      void refreshJob(id);
    }
  }, []);

  useEffect(() => {
    if (!jobId || job?.status === "completed" || job?.status === "failed") {
      return;
    }
    const timer = window.setInterval(() => {
      void refreshJob(jobId);
    }, 1000);
    return () => window.clearInterval(timer);
  }, [jobId, job?.status]);

  useEffect(() => {
    if (job?.status === "completed" && jobId) {
      void fetchResults(jobId);
      void fetchReport(jobId);
    }
  }, [job?.status, jobId]);

  useEffect(() => {
    setFocusedRegion(null);
  }, [selectedId]);

  const selectedStructure = useMemo(() => {
    if (!results?.structures.length) {
      return null;
    }
    return results.structures.find((item) => item.id === selectedId) ?? results.structures[0];
  }, [results, selectedId]);

  async function refreshJob(id = jobId) {
    if (!id) {
      return;
    }
    const response = await fetch(`${API_BASE}/api/jobs/${id}`);
    if (!response.ok) {
      return;
    }
    const data = await response.json();
    setJob(data.job);
    if (data.job.status === "completed") {
      await fetchResults(id);
    }
  }

  async function fetchResults(id: string) {
    const response = await fetch(`${API_BASE}/api/jobs/${id}/results`);
    if (!response.ok) {
      return;
    }
    const data = await response.json();
    setResults(data);
    if (!selectedId && data.structures?.length) {
      setSelectedId(data.structures[0].id);
    }
  }

  async function fetchReport(id: string) {
    const response = await fetch(`${API_BASE}/api/jobs/${id}/report.md`);
    if (!response.ok) {
      return;
    }
    setReportMarkdown(await response.text());
  }

  function onFilesChanged(event: ChangeEvent<HTMLInputElement>) {
    const nextFiles = Array.from(event.target.files ?? []);
    setFiles(nextFiles);
    setNotice(null);
  }

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!files.length) {
      setNotice("Select at least one structure file or zip archive.");
      return;
    }
    setSubmitting(true);
    setNotice(null);
    setResults(null);
    setReportMarkdown("");
    setSelectedId(null);
    setFocusedRegion(null);

    const formData = new FormData();
    files.forEach((file) => formData.append("files", file));
    formData.append("query", query);
    formData.append("profile", profile);
    formData.append("rank_by", rankBy);
    formData.append("glob_pattern", "*");
    formData.append("chain_a", chainA);
    formData.append("chain_b", chainB);
    formData.append("max_workers", String(maxWorkers));
    formData.append("enable_llm_synthesis", String(enableLlmSynthesis));
    formData.append("llm_provider", llmProvider);
    formData.append("llm_model", llmModel);
    formData.append("llm_base_url", llmBaseUrl);
    formData.append("llm_api_key", llmApiKey);
    formData.append("llm_temperature", "0.2");

    try {
      const response = await fetch(`${API_BASE}/api/jobs`, {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.detail ?? "Job creation failed.");
      }
      const data = await response.json();
      setJobId(data.job_id);
      setJob(data.job);
      setNotice(`Job ${data.job_id} queued.`);
      await refreshJob(data.job_id);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Job creation failed.");
    } finally {
      setSubmitting(false);
    }
  }

  function selectEvidence(key: string) {
    setActiveEvidence(key);
    setFocusedRegion(null);
  }

  function selectReportRegion(region: FocusedRegion) {
    setActiveEvidence(region.evidenceKey);
    setFocusedRegion(region);
  }

  const progress = job?.total_count ? job.completed_count / job.total_count : 0;

  return (
    <div className="app-shell">
      <header className="topbar">
        <LogoMark />
        <div className="topbar-actions">
          <StatusPill online={apiOnline} />
          <button className="icon-button" onClick={() => void refreshJob()} title="Refresh job">
            <RefreshCw size={18} />
          </button>
        </div>
      </header>

      <main className="workspace">
        <section className="panel control-panel">
          <div className="panel-heading">
            <FileArchive size={19} />
            <h2>Batch Input</h2>
          </div>

          <form onSubmit={onSubmit} className="control-form">
            <input
              ref={fileInputRef}
              className="file-input"
              type="file"
              multiple
              accept=".pdb,.cif,.mmcif,.zip"
              onChange={onFilesChanged}
            />
            <button className="upload-drop" type="button" onClick={() => fileInputRef.current?.click()}>
              <Upload size={20} />
              <span>{files.length ? `${files.length} file(s) selected` : "Select PDB/CIF files"}</span>
            </button>

            {files.length > 0 && (
              <div className="file-list">
                {files.slice(0, 5).map((file) => (
                  <span key={`${file.name}-${file.size}`}>{file.name}</span>
                ))}
                {files.length > 5 && <span>+{files.length - 5} more</span>}
              </div>
            )}

            <label>
              Query
              <textarea value={query} onChange={(event) => setQuery(event.target.value)} rows={4} />
            </label>

            <label>
              Profile
              <select
                value={profile}
                onChange={(event) => {
                  setProfile(event.target.value);
                  const nextProfile = profiles.find((item) => item.name === event.target.value);
                  if (nextProfile) {
                    setRankBy(nextProfile.default_rank_by);
                  }
                }}
              >
                {profiles.length ? (
                  profiles.map((item) => (
                    <option key={item.name} value={item.name}>
                      {item.label}
                    </option>
                  ))
                ) : (
                  <option value="triage_default">Batch triage</option>
                )}
              </select>
            </label>

            <label>
              Rank by
              <select value={rankBy} onChange={(event) => setRankBy(event.target.value)}>
                {rankOptions.map((item) => (
                  <option key={item.value} value={item.value}>
                    {item.label}
                  </option>
                ))}
              </select>
            </label>

            <div className="inline-grid">
              <label>
                Chain A
                <input value={chainA} onChange={(event) => setChainA(event.target.value)} placeholder="auto" />
              </label>
              <label>
                Chain B
                <input value={chainB} onChange={(event) => setChainB(event.target.value)} placeholder="auto" />
              </label>
            </div>

            <label>
              Workers
              <input
                type="number"
                min={1}
                max={12}
                value={maxWorkers}
                onChange={(event) => setMaxWorkers(Number(event.target.value))}
              />
            </label>

            <label className="checkbox-row">
              <input
                type="checkbox"
                checked={enableLlmSynthesis}
                onChange={(event) => setEnableLlmSynthesis(event.target.checked)}
              />
              <span>LLM synthesis</span>
            </label>

            {enableLlmSynthesis && (
              <div className="llm-grid">
                <label>
                  Provider
                  <select
                    value={llmProvider}
                    onChange={(event) => {
                      const next = providerOptions.find((item) => item.value === event.target.value);
                      setLlmProvider(event.target.value);
                      if (next) {
                        setLlmModel(next.model);
                        setLlmBaseUrl(next.baseUrl);
                      }
                    }}
                  >
                    {providerOptions.map((item) => (
                      <option key={item.value} value={item.value}>
                        {item.label}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  Model
                  <input value={llmModel} onChange={(event) => setLlmModel(event.target.value)} />
                </label>
                <label className="wide-field">
                  Base URL
                  <input value={llmBaseUrl} onChange={(event) => setLlmBaseUrl(event.target.value)} />
                </label>
                <label className="wide-field">
                  API key
                  <input
                    type="password"
                    value={llmApiKey}
                    onChange={(event) => setLlmApiKey(event.target.value)}
                    placeholder="env or paste key"
                  />
                </label>
              </div>
            )}

            <button className="primary-button" type="submit" disabled={submitting || !apiOnline}>
              {submitting ? <Loader2 size={18} className="spin" /> : <Play size={18} />}
              <span>Run Batch</span>
            </button>
          </form>

          {notice && <p className="notice">{notice}</p>}

          <div className="job-strip">
            <div>
              <span className="muted">Status</span>
              <strong>{job?.status ?? "idle"}</strong>
            </div>
            <div>
              <span className="muted">Progress</span>
              <strong>
                {job ? `${job.completed_count}/${job.total_count || "?"}` : "0/0"}
              </strong>
            </div>
          </div>
          <div className="progress-track">
            <div className="progress-fill" style={{ width: `${Math.round(progress * 100)}%` }} />
          </div>
        </section>

        <section className="panel results-panel">
          <div className="panel-heading">
            <Activity size={19} />
            <h2>Ranked Structures</h2>
          </div>
          <RankingTable
            ranking={results?.ranking ?? []}
            structures={results?.structures ?? []}
            selectedId={selectedStructure?.id ?? null}
            onSelect={setSelectedId}
          />
          <MetricsGrid structure={selectedStructure} />
        </section>

        <section className="panel viewer-panel">
          <div className="panel-heading split">
            <span>
              <Microscope size={19} />
              <h2>{selectedStructure?.pdb_id ?? "Structure Viewer"}</h2>
            </span>
            {jobId && (
              <a className="icon-link" href={`${API_BASE}/api/jobs/${jobId}/report.md`} target="_blank">
                <ArrowDownToLine size={17} />
                <span>Report</span>
              </a>
            )}
          </div>
          <div className="viewer-body">
            <div className="viewer-column">
              <StructureViewer
                structure={selectedStructure}
                activeEvidence={activeEvidence}
                focusedRegion={focusedRegion}
              />
              <EvidenceControls structure={selectedStructure} activeEvidence={activeEvidence} onChange={selectEvidence} />
            </div>
            <ReportPanel
              markdown={reportMarkdown}
              selectedStructure={selectedStructure}
              onRegionSelect={selectReportRegion}
            />
          </div>
        </section>
      </main>
    </div>
  );
}

function LogoMark() {
  return (
    <div className="logo-mark">
      <div className="terminal-line">&gt; MIRA --load model:</div>
      <pre aria-hidden="true">{".--..--.\n  \\\\  //\n   \\\\//\n   //\\\\\n  //  \\\\"}</pre>
      <div>
        <div className="brand-word">MIRA</div>
        <div className="brand-subtitle">MOLECULAR INTELLIGENCE & REASONING AGENT</div>
      </div>
    </div>
  );
}

function StatusPill({ online }: { online: boolean | null }) {
  const label = online === null ? "checking" : online ? "api online" : "api offline";
  return (
    <span className={`status-pill ${online ? "ok" : online === false ? "bad" : ""}`}>
      <Server size={15} />
      {label}
    </span>
  );
}

function RankingTable({
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
    return <div className="empty-state">No batch results yet.</div>;
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

function MetricsGrid({ structure }: { structure: StructureResult | null }) {
  const metrics = Object.entries(structure?.metrics ?? {}).filter(([key]) => key !== "total_execution_time");
  if (!structure) {
    return <div className="empty-state compact">Select a completed structure to inspect metrics.</div>;
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

function StructureViewer({
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
      const response = await fetch(`${API_BASE}${renderedStructure.structure_url}`);
      const text = await response.text();
      if (cancelled || !viewerRef.current) {
        return;
      }
      viewerRef.current.innerHTML = "";
      const viewer = window.$3Dmol.createViewer(viewerRef.current, { backgroundColor: "#f8fbff" });
      const format = renderedStructure.filename.toLowerCase().endsWith(".cif") ? "cif" : "pdb";
      viewer.addModel(text, format);
      viewer.setStyle({}, { cartoon: { color: "spectrum", opacity: 0.82 } });
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
      viewer.render();
      setViewerState("ready");
    }

    renderStructure().catch(() => setViewerState("viewer error"));
    return () => {
      cancelled = true;
    };
  }, [structure?.id, activeEvidence, focusedRegion?.evidenceKey, focusedRegion?.chain, focusedRegion?.residueNumber]);

  return (
    <div className="viewer-frame">
      <div ref={viewerRef} className="viewer-canvas" />
      {viewerState !== "ready" && <div className="viewer-state">{viewerState}</div>}
    </div>
  );
}

function EvidenceControls({
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
      {structure?.summary && <p className="summary-text">{structure.summary}</p>}
    </div>
  );
}

function ReportPanel({
  markdown,
  selectedStructure,
  onRegionSelect,
}: {
  markdown: string;
  selectedStructure: StructureResult | null;
  onRegionSelect: (region: FocusedRegion) => void;
}) {
  const lines = markdown ? markdown.split("\n") : [];
  return (
    <div className="report-panel">
      <div className="report-heading">
        <FileText size={17} />
        <span>Synthesis Report</span>
      </div>
      {lines.length ? (
        <div className="report-scroll">
          {lines.map((line, index) => (
            <ReportLine key={`${line}-${index}`} line={line} onRegionSelect={onRegionSelect} />
          ))}
        </div>
      ) : (
        <div className="empty-state compact">Run a batch to generate the report.</div>
      )}
      {selectedStructure && (
        <div className="report-focus-note">
          Report region links highlight residues on <strong>{selectedStructure.pdb_id}</strong>.
        </div>
      )}
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
    const content = renderReportInline(line.slice(2), onRegionSelect);
    return <h3 className="report-title">{content}</h3>;
  }
  if (line.startsWith("## ")) {
    const content = renderReportInline(line.slice(3), onRegionSelect);
    return <h4 className="report-section">{content}</h4>;
  }
  if (line.startsWith("### ")) {
    const content = renderReportInline(line.slice(4), onRegionSelect);
    return <h5 className="report-subsection">{content}</h5>;
  }
  if (line.startsWith("#### ")) {
    const content = renderReportInline(line.slice(5), onRegionSelect);
    return <h6 className="report-minihead">{content}</h6>;
  }
  if (line.startsWith("|")) {
    return <pre className="report-table-line">{line}</pre>;
  }
  if (line.startsWith("- ")) {
    const content = renderReportInline(line.slice(2), onRegionSelect);
    return <div className="report-bullet">{content}</div>;
  }
  const content = renderReportInline(line, onRegionSelect);
  return <p className="report-paragraph">{content}</p>;
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
