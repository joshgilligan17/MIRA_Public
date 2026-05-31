import {
  Activity,
  ArrowDownToLine,
  FileArchive,
  Folder,
  FolderPlus,
  Loader2,
  MessageSquare,
  Microscope,
  Play,
  RefreshCw,
  SlidersHorizontal,
  Upload,
} from "lucide-react";
import { ChangeEvent, FormEvent, ReactNode, useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  BrowserRouter,
  Link,
  Navigate,
  NavLink,
  Route,
  Routes,
  useLocation,
  useNavigate,
  useParams,
} from "react-router-dom";
import {
  API_BASE,
  ChatMessage,
  FocusedRegion,
  Job,
  Profile,
  Project,
  Results,
  StructureResult,
  SynthesisStatus,
  createProject,
  createProjectJob,
  getHealth,
  getJob,
  getProfiles,
  getProject,
  getProjectChat,
  getReport,
  getResults,
  listProjectJobs,
  listProjects,
  providerOptions,
  rankOptions,
  sendProjectChat,
  updateProject,
  uploadProjectStructure,
  uploadProjectTarget,
} from "./api";
import {
  EvidenceControls,
  LogoMark,
  MetricsGrid,
  RankingTable,
  RenderedMarkdown,
  ReportPanel,
  StatusPill,
  StructureViewer,
} from "./components";

export default function App() {
  return (
    <BrowserRouter>
      <MiraWorkspace />
    </BrowserRouter>
  );
}

function MiraWorkspace() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [profiles, setProfiles] = useState<Profile[]>([]);
  const [apiOnline, setApiOnline] = useState<boolean | null>(null);
  const [synthesisStatus, setSynthesisStatus] = useState<SynthesisStatus | null>(null);

  const refreshProjects = useCallback(async () => {
    const nextProjects = await listProjects();
    setProjects(nextProjects);
  }, []);

  useEffect(() => {
    getHealth()
      .then((health) => {
        setApiOnline(true);
        setSynthesisStatus(health.synthesis ?? null);
      })
      .catch(() => {
        setApiOnline(false);
        setSynthesisStatus(null);
      });
    getProfiles()
      .then(setProfiles)
      .catch(() => setProfiles([]));
    refreshProjects().catch(() => setProjects([]));
  }, [refreshProjects]);

  return (
    <div className="app-shell">
      <Sidebar projects={projects} apiOnline={apiOnline} synthesisStatus={synthesisStatus} />
      <main className="workspace-main">
        <Routes>
          <Route path="/" element={<Navigate to="/projects" replace />} />
          <Route path="/projects" element={<ProjectsPage projects={projects} refreshProjects={refreshProjects} />} />
          <Route path="/projects/:projectId" element={<ProjectIndex />} />
          <Route
            path="/projects/:projectId/chat"
            element={<ChatPage refreshProjects={refreshProjects} />}
          />
          <Route
            path="/projects/:projectId/batch"
            element={<BatchPage profiles={profiles} refreshProjects={refreshProjects} />}
          />
          <Route
            path="/projects/:projectId/jobs/:jobId"
            element={<BatchPage profiles={profiles} refreshProjects={refreshProjects} />}
          />
          <Route path="*" element={<Navigate to="/projects" replace />} />
        </Routes>
      </main>
    </div>
  );
}

function Sidebar({
  projects,
  apiOnline,
  synthesisStatus,
}: {
  projects: Project[];
  apiOnline: boolean | null;
  synthesisStatus: SynthesisStatus | null;
}) {
  const location = useLocation();
  const activeProjectId = location.pathname.match(/\/projects\/([^/]+)/)?.[1] ?? null;
  const activeProject = projects.find((project) => project.id === activeProjectId) ?? null;
  const resultsHref = activeProject?.selected_job_id
    ? `/projects/${activeProject.id}/jobs/${activeProject.selected_job_id}`
    : activeProject
      ? `/projects/${activeProject.id}/batch`
      : "/projects";

  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <LogoMark compact />
      </div>
      <StatusPill online={apiOnline} synthesis={synthesisStatus} />
      <nav className="mode-nav">
        <NavLink to="/projects" end className={({ isActive }) => (isActive ? "nav-item active" : "nav-item")}>
          <Folder size={18} />
          <span>Projects</span>
        </NavLink>
        {activeProject && (
          <>
            <NavLink
              to={`/projects/${activeProject.id}/chat`}
              className={({ isActive }) => (isActive ? "nav-item active" : "nav-item")}
            >
              <MessageSquare size={18} />
              <span>Chat</span>
            </NavLink>
            <NavLink
              to={`/projects/${activeProject.id}/batch`}
              className={({ isActive }) => (isActive ? "nav-item active" : "nav-item")}
            >
              <FileArchive size={18} />
              <span>Batch</span>
            </NavLink>
            <NavLink to={resultsHref} className={({ isActive }) => (isActive ? "nav-item active" : "nav-item")}>
              <Activity size={18} />
              <span>Results</span>
            </NavLink>
          </>
        )}
      </nav>
      <div className="sidebar-section">
        <div className="sidebar-section-title">Folders</div>
        <div className="project-switcher">
          {projects.slice(0, 8).map((project) => (
            <Link
              key={project.id}
              to={`/projects/${project.id}/chat`}
              className={project.id === activeProjectId ? "project-link active" : "project-link"}
            >
              <span>{project.name}</span>
              <small>{project.job_count} batch runs</small>
            </Link>
          ))}
          {!projects.length && <div className="sidebar-empty">No projects yet.</div>}
        </div>
      </div>
    </aside>
  );
}

function ProjectIndex() {
  const { projectId } = useParams();
  return <Navigate to={`/projects/${projectId}/chat`} replace />;
}

function ProjectsPage({
  projects,
  refreshProjects,
}: {
  projects: Project[];
  refreshProjects: () => Promise<void>;
}) {
  const navigate = useNavigate();
  const [name, setName] = useState("New MIRA project");
  const [description, setDescription] = useState("");
  const [creating, setCreating] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);

  async function onCreate(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setCreating(true);
    setNotice(null);
    try {
      const project = await createProject(name, description);
      await refreshProjects();
      window.localStorage.setItem("mira:lastProjectId", project.id);
      navigate(`/projects/${project.id}/chat`);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Project creation failed.");
    } finally {
      setCreating(false);
    }
  }

  return (
    <div className="page-stack">
      <header className="page-header">
        <div>
          <p className="eyebrow">MIRA workspace</p>
          <h1>Projects</h1>
        </div>
        <form className="new-project-form" onSubmit={onCreate}>
          <input value={name} onChange={(event) => setName(event.target.value)} />
          <input
            value={description}
            onChange={(event) => setDescription(event.target.value)}
            placeholder="Target, campaign, or design goal"
          />
          <button className="primary-button" type="submit" disabled={creating}>
            {creating ? <Loader2 size={17} className="spin" /> : <FolderPlus size={17} />}
            <span>Create</span>
          </button>
        </form>
      </header>
      {notice && <p className="notice">{notice}</p>}
      <section className="project-grid">
        {projects.map((project) => (
          <Link className="project-card" to={`/projects/${project.id}/chat`} key={project.id}>
            <div>
              <h2>{project.name}</h2>
              <p>{project.description || "Structure reasoning workspace"}</p>
            </div>
            <div className="project-card-meta">
              <span>{project.target_original_name || "No target"}</span>
              <span>{project.job_count} batch runs</span>
            </div>
          </Link>
        ))}
        {!projects.length && <div className="empty-state">Create a project to begin.</div>}
      </section>
    </div>
  );
}

function ChatPage({ refreshProjects }: { refreshProjects: () => Promise<void> }) {
  const { projectId = "" } = useParams();
  const targetInputRef = useRef<HTMLInputElement>(null);
  const structureInputRef = useRef<HTMLInputElement>(null);
  const [project, setProject] = useState<Project | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [selectedStructure, setSelectedStructure] = useState<StructureResult | null>(null);
  const [reportMarkdown, setReportMarkdown] = useState("");
  const [prompt, setPrompt] = useState("");
  const [notice, setNotice] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [uploadingTarget, setUploadingTarget] = useState(false);
  const [uploadingStructure, setUploadingStructure] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [activeEvidence, setActiveEvidence] = useState("interface_residues");
  const [focusedRegion, setFocusedRegion] = useState<FocusedRegion | null>(null);

  const loadChatProject = useCallback(async () => {
    if (!projectId) {
      return;
    }
    setLoading(true);
    try {
      const [nextProject, nextMessages] = await Promise.all([getProject(projectId), getProjectChat(projectId)]);
      setProject(nextProject);
      setMessages(nextMessages);
      await loadSelectedProjectStructure(nextProject, setSelectedStructure, setReportMarkdown);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Project load failed.");
    } finally {
      setLoading(false);
    }
  }, [projectId]);

  useEffect(() => {
    void loadChatProject();
  }, [loadChatProject]);

  async function onTargetChanged(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file || !projectId) {
      return;
    }
    setUploadingTarget(true);
    setNotice(null);
    try {
      const nextProject = await uploadProjectTarget(projectId, file);
      setProject(nextProject);
      setSelectedStructure(nextProject.target_structure ?? null);
      setReportMarkdown("");
      await refreshProjects();
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Target upload failed.");
    } finally {
      setUploadingTarget(false);
      event.target.value = "";
    }
  }

  async function onStructureChanged(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file || !projectId) {
      return;
    }
    setUploadingStructure(true);
    setNotice(null);
    try {
      const response = await uploadProjectStructure(projectId, file);
      setProject(response.project);
      setSelectedStructure(response.structure);
      setReportMarkdown("");
      setFocusedRegion(null);
      await refreshProjects();
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Structure upload failed.");
    } finally {
      setUploadingStructure(false);
      event.target.value = "";
    }
  }

  async function onSelectChatStructure(value: string) {
    if (!project) {
      return;
    }
    const structure = findProjectStructure(project, value);
    if (!structure) {
      return;
    }
    setSelectedStructure(structure);
    setReportMarkdown("");
    setFocusedRegion(null);
    try {
      const nextProject = await updateProject(project.id, {
        selected_job_id: null,
        selected_structure_id: structure.id,
      });
      setProject(nextProject);
      await refreshProjects();
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Structure selection failed.");
    }
  }

  async function onSend(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!prompt.trim() || !project) {
      return;
    }
    const message = prompt.trim();
    setPrompt("");
    setSubmitting(true);
    setNotice(null);
    try {
      const chatResponse = await sendProjectChat(
        project.id,
        message,
        project.selected_job_id,
        selectedStructure?.id ?? project.selected_structure_id,
      );
      setMessages(chatResponse.messages);
      const nextProject = chatResponse.project ?? (await getProject(project.id));
      setProject(nextProject);
      await loadSelectedProjectStructure(nextProject, setSelectedStructure, setReportMarkdown);
      await refreshProjects();
    } catch (error) {
      setPrompt(message);
      setNotice(error instanceof Error ? error.message : "Chat failed.");
    } finally {
      setSubmitting(false);
    }
  }

  const inspectorStructure = selectedStructure ?? project?.target_structure ?? null;
  const chatStructures = projectStructures(project);
  const selectedStructureValue = inspectorStructure?.id ?? "";

  return (
    <div className="project-layout">
      <section className="center-column chat-column">
        <PageTitle
          eyebrow={project?.name || "Project"}
          title="Chat"
          action={
            project && (
              <Link className="secondary-button" to={`/projects/${project.id}/batch`}>
                <FileArchive size={17} />
                <span>Analyze candidate binders</span>
              </Link>
            )
          }
        />
        <div className="chat-toolbar">
          <input
            ref={targetInputRef}
            className="file-input"
            type="file"
            accept=".pdb,.cif,.mmcif"
            onChange={onTargetChanged}
          />
          <input
            ref={structureInputRef}
            className="file-input"
            type="file"
            accept=".pdb,.cif,.mmcif"
            onChange={onStructureChanged}
          />
          <label className="structure-select-label">
            Structure
            <select value={selectedStructureValue} onChange={(event) => void onSelectChatStructure(event.target.value)}>
              {!chatStructures.length && <option value="">No structure selected</option>}
              {chatStructures.map((structure) => (
                <option key={structure.id} value={structure.id}>
                  {structure.id === "target" ? `Target: ${structure.pdb_id}` : structure.pdb_id}
                </option>
              ))}
              {inspectorStructure && !chatStructures.some((structure) => structure.id === inspectorStructure.id) && (
                <option value={inspectorStructure.id}>{inspectorStructure.pdb_id}</option>
              )}
            </select>
          </label>
          <button className="secondary-button" type="button" onClick={() => targetInputRef.current?.click()}>
            {uploadingTarget ? <Loader2 size={17} className="spin" /> : <Upload size={17} />}
            <span>{project?.target_original_name || "Upload target"}</span>
          </button>
          <button className="secondary-button" type="button" onClick={() => structureInputRef.current?.click()}>
            {uploadingStructure ? <Loader2 size={17} className="spin" /> : <Upload size={17} />}
            <span>Upload structure</span>
          </button>
          <button className="icon-button" type="button" onClick={() => void loadChatProject()} title="Refresh project">
            <RefreshCw size={18} />
          </button>
        </div>
        {notice && <p className="notice">{notice}</p>}
        <div className="chat-messages">
          {messages.map((message) => (
            <ChatBubble key={message.id} message={message} onRegionSelect={setFocusedRegion} />
          ))}
          {!messages.length && !loading && <div className="empty-state">No chat messages yet.</div>}
          {loading && <div className="empty-state">Loading project.</div>}
        </div>
        <form className="composer" onSubmit={onSend}>
          <textarea
            value={prompt}
            onChange={(event) => setPrompt(event.target.value)}
            rows={3}
            placeholder="Ask about the target, selected candidate, report, or highlighted regions"
          />
          <button className="primary-button" type="submit" disabled={submitting || !prompt.trim()}>
            {submitting ? <Loader2 size={18} className="spin" /> : <MessageSquare size={18} />}
            <span>Send</span>
          </button>
        </form>
      </section>
      <StructureInspector
        structure={inspectorStructure}
        title={inspectorStructure?.pdb_id ?? "Structure"}
        activeEvidence={activeEvidence}
        focusedRegion={focusedRegion}
        reportMarkdown={reportMarkdown}
        reportHref={project?.selected_job_id ? `${API_BASE}/api/jobs/${project.selected_job_id}/report.md` : null}
        onEvidenceChange={(key) => {
          setActiveEvidence(key);
          setFocusedRegion(null);
        }}
        onRegionSelect={(region) => {
          setActiveEvidence(region.evidenceKey);
          setFocusedRegion(region);
        }}
      />
    </div>
  );
}

function BatchPage({
  profiles,
  refreshProjects,
}: {
  profiles: Profile[];
  refreshProjects: () => Promise<void>;
}) {
  const { projectId = "", jobId: routeJobId } = useParams();
  const navigate = useNavigate();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [project, setProject] = useState<Project | null>(null);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [selectedJobId, setSelectedJobId] = useState<string | null>(routeJobId ?? null);
  const [job, setJob] = useState<Job | null>(null);
  const [results, setResults] = useState<Results | null>(null);
  const [reportMarkdown, setReportMarkdown] = useState("");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [activeEvidence, setActiveEvidence] = useState("interface_residues");
  const [focusedRegion, setFocusedRegion] = useState<FocusedRegion | null>(null);
  const [files, setFiles] = useState<File[]>([]);
  const [query, setQuery] = useState("Rank candidate binders for this project target.");
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
  const [submitting, setSubmitting] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);

  const loadJobBundle = useCallback(
    async (nextJobId: string, preferredStructureId?: string | null) => {
      const status = await getJob(nextJobId);
      setJob(status.job);
      if (status.job.status === "completed") {
        const nextResults = await getResults(nextJobId);
        setResults(nextResults);
        setSelectedId((current) => {
          const candidate = preferredStructureId || current;
          if (candidate && nextResults.structures.some((item) => item.id === candidate)) {
            return candidate;
          }
          return nextResults.structures[0]?.id ?? null;
        });
        getReport(nextJobId)
          .then(setReportMarkdown)
          .catch(() => setReportMarkdown(""));
      }
    },
    [],
  );

  const loadBatchProject = useCallback(async () => {
    if (!projectId) {
      return;
    }
    try {
      const [nextProject, nextJobs] = await Promise.all([getProject(projectId), listProjectJobs(projectId)]);
      setProject(nextProject);
      setJobs(nextJobs);
      const nextJobId = routeJobId || nextProject.selected_job_id || nextJobs[0]?.id || null;
      setSelectedJobId(nextJobId);
      if (nextJobId) {
        await loadJobBundle(nextJobId, nextProject.selected_structure_id);
      } else {
        setJob(null);
        setResults(null);
        setReportMarkdown("");
      }
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Batch workspace load failed.");
    }
  }, [loadJobBundle, projectId, routeJobId]);

  useEffect(() => {
    void loadBatchProject();
  }, [loadBatchProject]);

  useEffect(() => {
    if (!selectedJobId || job?.status === "completed" || job?.status === "failed") {
      return;
    }
    const timer = window.setInterval(() => {
      void loadJobBundle(selectedJobId, selectedId);
    }, 1200);
    return () => window.clearInterval(timer);
  }, [job?.status, loadJobBundle, selectedId, selectedJobId]);

  const selectedStructure = useMemo(() => {
    if (!results?.structures.length) {
      return project?.target_structure ?? null;
    }
    return results.structures.find((item) => item.id === selectedId) ?? results.structures[0];
  }, [project?.target_structure, results, selectedId]);

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!files.length || !projectId) {
      setNotice("Select at least one candidate structure or zip archive.");
      return;
    }
    setSubmitting(true);
    setNotice(null);
    setFocusedRegion(null);
    try {
      const created = await createProjectJob(projectId, {
        files,
        query,
        profile,
        rankBy,
        chainA,
        chainB,
        maxWorkers,
        enableLlmSynthesis,
        llmProvider,
        llmModel,
        llmBaseUrl,
        llmApiKey,
      });
      setFiles([]);
      setJob(created.job);
      setSelectedJobId(created.job_id);
      setResults(null);
      setReportMarkdown("");
      await refreshProjects();
      navigate(`/projects/${projectId}/jobs/${created.job_id}`);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Job creation failed.");
    } finally {
      setSubmitting(false);
    }
  }

  async function onSelectStructure(id: string) {
    setSelectedId(id);
    setFocusedRegion(null);
    if (projectId && selectedJobId) {
      updateProject(projectId, { selected_job_id: selectedJobId, selected_structure_id: id })
        .then(setProject)
        .then(() => refreshProjects())
        .catch(() => undefined);
    }
  }

  const progress = job?.total_count ? job.completed_count / job.total_count : 0;

  return (
    <div className="project-layout">
      <section className="center-column batch-column">
        <PageTitle
          eyebrow={project?.name || "Project"}
          title="Batch"
          action={
            project && (
              <Link className="secondary-button" to={`/projects/${project.id}/chat`}>
                <MessageSquare size={17} />
                <span>Chat</span>
              </Link>
            )
          }
        />
        <form className="batch-runner" onSubmit={onSubmit}>
          <input
            ref={fileInputRef}
            className="file-input"
            type="file"
            multiple
            accept=".pdb,.cif,.mmcif,.zip"
            onChange={(event) => setFiles(Array.from(event.target.files ?? []))}
          />
          <button className="upload-drop" type="button" onClick={() => fileInputRef.current?.click()}>
            <Upload size={20} />
            <span>{files.length ? `${files.length} file(s) selected` : "Select candidates"}</span>
          </button>
          {files.length > 0 && (
            <div className="file-list">
              {files.slice(0, 5).map((file) => (
                <span key={`${file.name}-${file.size}`}>{file.name}</span>
              ))}
              {files.length > 5 && <span>+{files.length - 5} more</span>}
            </div>
          )}
          <textarea value={query} onChange={(event) => setQuery(event.target.value)} rows={3} />
          <div className="run-settings">
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
            <button className="primary-button" type="submit" disabled={submitting}>
              {submitting ? <Loader2 size={18} className="spin" /> : <Play size={18} />}
              <span>Run Batch</span>
            </button>
          </div>
          <details className="advanced-panel">
            <summary>
              <SlidersHorizontal size={16} />
              <span>Advanced</span>
            </summary>
            <div className="advanced-grid">
              <label>
                Chain A
                <input value={chainA} onChange={(event) => setChainA(event.target.value)} placeholder="auto" />
              </label>
              <label>
                Chain B
                <input value={chainB} onChange={(event) => setChainB(event.target.value)} placeholder="auto" />
              </label>
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
                <>
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
                      placeholder="server env by default"
                    />
                  </label>
                </>
              )}
            </div>
          </details>
        </form>
        {notice && <p className="notice">{notice}</p>}
        <section className="job-overview">
          <div className="job-strip">
            <div>
              <span className="muted">Status</span>
              <strong>{job?.status ?? "idle"}</strong>
            </div>
            <div>
              <span className="muted">Progress</span>
              <strong>{job ? `${job.completed_count}/${job.total_count || "?"}` : "0/0"}</strong>
            </div>
          </div>
          <div className="progress-track">
            <div className="progress-fill" style={{ width: `${Math.round(progress * 100)}%` }} />
          </div>
        </section>
        <section className="results-section">
          <div className="section-heading">
            <Activity size={18} />
            <h2>Ranked structures</h2>
          </div>
          <RankingTable
            ranking={results?.ranking ?? []}
            structures={results?.structures ?? []}
            selectedId={selectedStructure?.id ?? null}
            onSelect={onSelectStructure}
          />
        </section>
        {!!jobs.length && (
          <section className="job-list">
            {jobs.slice(0, 6).map((item) => (
              <button
                key={item.id}
                type="button"
                className={item.id === selectedJobId ? "job-list-item active" : "job-list-item"}
                onClick={() => navigate(`/projects/${projectId}/jobs/${item.id}`)}
              >
                <span>{item.id}</span>
                <strong>{item.status}</strong>
              </button>
            ))}
          </section>
        )}
      </section>
      <StructureInspector
        structure={selectedStructure}
        title={selectedStructure?.pdb_id ?? "Structure"}
        activeEvidence={activeEvidence}
        focusedRegion={focusedRegion}
        reportMarkdown={reportMarkdown}
        reportHref={selectedJobId ? `${API_BASE}/api/jobs/${selectedJobId}/report.md` : null}
        onEvidenceChange={(key) => {
          setActiveEvidence(key);
          setFocusedRegion(null);
        }}
        onRegionSelect={(region) => {
          setActiveEvidence(region.evidenceKey);
          setFocusedRegion(region);
        }}
      />
    </div>
  );
}

function StructureInspector({
  structure,
  title,
  activeEvidence,
  focusedRegion,
  reportMarkdown,
  reportHref,
  onEvidenceChange,
  onRegionSelect,
}: {
  structure: StructureResult | null;
  title: string;
  activeEvidence: string;
  focusedRegion: FocusedRegion | null;
  reportMarkdown: string;
  reportHref: string | null;
  onEvidenceChange: (key: string) => void;
  onRegionSelect: (region: FocusedRegion) => void;
}) {
  return (
    <aside className="inspector-column">
      <div className="inspector-heading">
        <span>
          <Microscope size={19} />
          <h2>{title}</h2>
        </span>
        {reportHref && (
          <a className="icon-link" href={reportHref} target="_blank">
            <ArrowDownToLine size={17} />
            <span>Report</span>
          </a>
        )}
      </div>
      <StructureViewer structure={structure} activeEvidence={activeEvidence} focusedRegion={focusedRegion} />
      <EvidenceControls structure={structure} activeEvidence={activeEvidence} onChange={onEvidenceChange} />
      <MetricsGrid structure={structure} />
      <ReportPanel markdown={reportMarkdown} selectedStructure={structure} onRegionSelect={onRegionSelect} />
    </aside>
  );
}

function ChatBubble({
  message,
  onRegionSelect,
}: {
  message: ChatMessage;
  onRegionSelect: (region: FocusedRegion) => void;
}) {
  const toolEvents = message.tool_events ?? [];
  return (
    <article className={`chat-bubble ${message.role}`}>
      <div className="chat-role">{message.role === "assistant" ? "MIRA" : "You"}</div>
      {toolEvents.length > 0 && (
        <div className="tool-events">
          {toolEvents.map((event, index) => (
            <span className={event.success ? "tool-event ok" : "tool-event bad"} key={`${event.tool}-${index}`}>
              {event.success ? "Done" : "Issue"}: {event.tool.replace(/_/g, " ")}
            </span>
          ))}
        </div>
      )}
      <RenderedMarkdown markdown={message.content} onRegionSelect={onRegionSelect} />
    </article>
  );
}

function PageTitle({
  eyebrow,
  title,
  action,
}: {
  eyebrow: string;
  title: string;
  action?: ReactNode;
}) {
  return (
    <header className="page-header compact">
      <div>
        <p className="eyebrow">{eyebrow}</p>
        <h1>{title}</h1>
      </div>
      {action}
    </header>
  );
}

async function loadSelectedProjectStructure(
  project: Project,
  setSelectedStructure: (structure: StructureResult | null) => void,
  setReportMarkdown: (markdown: string) => void,
) {
  const localStructure = findProjectStructure(project, project.selected_structure_id || "target");
  if (localStructure && !project.selected_job_id) {
    setSelectedStructure(localStructure);
    setReportMarkdown("");
    return;
  }
  if (!project.selected_job_id) {
    setSelectedStructure(localStructure ?? projectStructures(project)[0] ?? null);
    setReportMarkdown("");
    return;
  }
  try {
    const [results, report] = await Promise.all([
      getResults(project.selected_job_id),
      getReport(project.selected_job_id).catch(() => ""),
    ]);
    const selected =
      localStructure ??
      results.structures.find((item) => item.id === project.selected_structure_id) ?? results.structures[0] ?? null;
    setSelectedStructure(selected ?? project.target_structure ?? null);
    setReportMarkdown(localStructure ? "" : report);
  } catch {
    setSelectedStructure(localStructure ?? project.target_structure ?? null);
    setReportMarkdown("");
  }
}

function projectStructures(project: Project | null): StructureResult[] {
  if (!project) {
    return [];
  }
  return [project.target_structure, ...(project.structures ?? [])].filter(Boolean) as StructureResult[];
}

function findProjectStructure(project: Project, structureId?: string | null): StructureResult | null {
  if (!structureId) {
    return null;
  }
  return projectStructures(project).find((structure) => structure.id === structureId) ?? null;
}
