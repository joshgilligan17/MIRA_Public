import {
  Activity,
  ArrowDownToLine,
  FileArchive,
  Folder,
  FolderPlus,
  Home,
  Loader2,
  MessageSquare,
  Microscope,
  PanelLeftClose,
  PanelLeftOpen,
  Play,
  RefreshCw,
  SlidersHorizontal,
  Trash2,
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
  ProjectDesignRun,
  Results,
  StructureResult,
  SynthesisStatus,
  createProject,
  createProjectJob,
  deleteProject,
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
  const [sidebarCollapsed, setSidebarCollapsed] = useState(() => {
    try {
      return window.localStorage.getItem("mira:sidebarCollapsed") === "true";
    } catch {
      return false;
    }
  });

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

  useEffect(() => {
    try {
      window.localStorage.setItem("mira:sidebarCollapsed", sidebarCollapsed ? "true" : "false");
    } catch {
      // Ignore storage failures; the toggle should still work for the session.
    }
  }, [sidebarCollapsed]);

  return (
    <div className={sidebarCollapsed ? "app-shell sidebar-collapsed" : "app-shell"}>
      <Sidebar
        projects={projects}
        apiOnline={apiOnline}
        synthesisStatus={synthesisStatus}
        collapsed={sidebarCollapsed}
        onToggleCollapsed={() => setSidebarCollapsed((value) => !value)}
      />
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
            path="/projects/:projectId/workspace"
            element={<BatchPage profiles={profiles} refreshProjects={refreshProjects} />}
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
  collapsed,
  onToggleCollapsed,
}: {
  projects: Project[];
  apiOnline: boolean | null;
  synthesisStatus: SynthesisStatus | null;
  collapsed: boolean;
  onToggleCollapsed: () => void;
}) {
  const location = useLocation();
  const activeProjectId = location.pathname.match(/\/projects\/([^/]+)/)?.[1] ?? null;
  const activeProject = projects.find((project) => project.id === activeProjectId) ?? null;
  const workspaceActive = Boolean(activeProject && /\/projects\/[^/]+\/(workspace|batch|jobs)/.test(location.pathname));

  return (
    <aside className={collapsed ? "sidebar collapsed" : "sidebar"}>
      <div className="sidebar-header">
        {!collapsed && (
          <div className="sidebar-brand">
            <LogoMark compact />
          </div>
        )}
        <button
          className="icon-button sidebar-toggle"
          type="button"
          onClick={onToggleCollapsed}
          title={collapsed ? "Show sidebar" : "Hide sidebar"}
          aria-label={collapsed ? "Show sidebar" : "Hide sidebar"}
        >
          {collapsed ? <PanelLeftOpen size={18} /> : <PanelLeftClose size={18} />}
        </button>
      </div>
      {!collapsed && (
        <>
          <StatusPill online={apiOnline} synthesis={synthesisStatus} />
          <nav className="mode-nav">
            <NavLink to="/projects" end className={({ isActive }) => (isActive ? "nav-item active" : "nav-item")}>
              <Home size={18} />
              <span>Home</span>
            </NavLink>
            {activeProject && (
              <>
                <div className="project-nav-heading">
                  <span>Project</span>
                  <strong>{activeProject.name}</strong>
                </div>
                <NavLink
                  to={`/projects/${activeProject.id}/chat`}
                  className={({ isActive }) => (isActive ? "nav-item active" : "nav-item")}
                >
                  <MessageSquare size={18} />
                  <span>Chat</span>
                </NavLink>
                <NavLink
                  to={`/projects/${activeProject.id}/workspace`}
                  className={() => (workspaceActive ? "nav-item active" : "nav-item")}
                >
                  <FileArchive size={18} />
                  <span>Workspace</span>
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
                  <small>{project.design_run_ids?.length ?? 0} generations</small>
                </Link>
              ))}
              {!projects.length && <div className="sidebar-empty">No projects yet.</div>}
            </div>
          </div>
        </>
      )}
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
  const [deletingProjectId, setDeletingProjectId] = useState<string | null>(null);
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

  async function onDeleteProject(project: Project) {
    const confirmed = window.confirm(`Delete "${project.name}" and its uploaded project files?`);
    if (!confirmed) {
      return;
    }
    setDeletingProjectId(project.id);
    setNotice(null);
    try {
      await deleteProject(project.id);
      if (window.localStorage.getItem("mira:lastProjectId") === project.id) {
        window.localStorage.removeItem("mira:lastProjectId");
      }
      await refreshProjects();
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Project deletion failed.");
    } finally {
      setDeletingProjectId(null);
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
          <article className="project-card" key={project.id}>
            <Link className="project-card-main" to={`/projects/${project.id}/chat`}>
              <div>
                <h2>{project.name}</h2>
                <p>{project.description || "Structure reasoning workspace"}</p>
              </div>
              <div className="project-card-meta">
                <span>{project.target_original_name || "No target"}</span>
                <span>{project.job_count} screens</span>
              </div>
            </Link>
            <div className="project-card-actions">
              <button
                className="icon-button danger"
                type="button"
                onClick={() => void onDeleteProject(project)}
                disabled={deletingProjectId === project.id}
                title={`Delete ${project.name}`}
              >
                {deletingProjectId === project.id ? <Loader2 size={17} className="spin" /> : <Trash2 size={17} />}
              </button>
            </div>
          </article>
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

  const loadChatProject = useCallback(async (silent = false) => {
    if (!projectId) {
      return;
    }
    if (!silent) {
      setLoading(true);
    }
    try {
      const [nextProject, nextMessages] = await Promise.all([getProject(projectId), getProjectChat(projectId)]);
      setProject(nextProject);
      setMessages(nextMessages);
      await loadSelectedProjectStructure(nextProject, setSelectedStructure, setReportMarkdown);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Project load failed.");
    } finally {
      if (!silent) {
        setLoading(false);
      }
    }
  }, [projectId]);

  useEffect(() => {
    void loadChatProject();
  }, [loadChatProject]);

  const activeDesignRun = (project?.design_runs ?? []).some((run) =>
    ["preparing", "queued", "running"].includes(run.status),
  );

  useEffect(() => {
    if (!activeDesignRun) {
      return;
    }
    const timer = window.setInterval(() => {
      void loadChatProject(true);
    }, 2000);
    return () => window.clearInterval(timer);
  }, [activeDesignRun, loadChatProject]);

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
              <Link className="secondary-button" to={`/projects/${project.id}/workspace`}>
                <FileArchive size={17} />
                <span>Open workspace</span>
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
  const [query, setQuery] = useState("Filter and rank candidate binders for this project target.");
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
      const nextJobId = routeJobId || nextProject.selected_job_id || null;
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
    const localStructure = project ? findProjectStructure(project, project.selected_structure_id || "target") : null;
    if (!selectedJobId || !results?.structures.length) {
      return localStructure ?? project?.target_structure ?? null;
    }
    return results.structures.find((item) => item.id === selectedId) ?? results.structures[0] ?? localStructure;
  }, [project, results, selectedId, selectedJobId]);
  const designRuns = project?.design_runs ?? [];
  const activeDesignRun = designRuns.some((run) => ["preparing", "queued", "running"].includes(run.status));

  useEffect(() => {
    if (!activeDesignRun) {
      return;
    }
    const timer = window.setInterval(() => {
      void loadBatchProject();
    }, 2000);
    return () => window.clearInterval(timer);
  }, [activeDesignRun, loadBatchProject]);

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

  async function onSelectProjectStructure(id: string) {
    if (!projectId) {
      return;
    }
    setSelectedJobId(null);
    setJob(null);
    setResults(null);
    setSelectedId(null);
    setReportMarkdown("");
    setFocusedRegion(null);
    setProject((current) => (current ? { ...current, selected_job_id: null, selected_structure_id: id } : current));
    try {
      const nextProject = await updateProject(projectId, { selected_job_id: null, selected_structure_id: id });
      setProject(nextProject);
      await refreshProjects();
      if (routeJobId) {
        navigate(`/projects/${projectId}/workspace`);
      }
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Structure selection failed.");
    }
  }

  const progress = job?.total_count ? job.completed_count / job.total_count : 0;

  return (
    <div className="project-layout">
      <section className="center-column batch-column">
        <PageTitle
          eyebrow={project?.name || "Project"}
          title="Workspace"
          action={
            project && (
              <Link className="secondary-button" to={`/projects/${project.id}/chat`}>
                <MessageSquare size={17} />
                <span>Chat</span>
              </Link>
            )
          }
        />
        <WorkspacePanel project={project} designRuns={designRuns} jobs={jobs} projectId={projectId} />
        <ProjectStructureLibrary
          structures={projectStructures(project)}
          selectedId={!selectedJobId ? selectedStructure?.id ?? null : null}
          onSelect={(id) => void onSelectProjectStructure(id)}
        />
        <form className="batch-runner" onSubmit={onSubmit}>
          <div className="section-heading">
            <FileArchive size={18} />
            <h2>Screen candidate structures</h2>
          </div>
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
            <h2>Top filtered designs</h2>
          </div>
          <RankingTable
            ranking={results?.ranking ?? []}
            structures={results?.structures ?? []}
            selectedId={selectedStructure?.id ?? null}
            onSelect={onSelectStructure}
          />
          {!results?.ranking.length && (
            <div className="empty-state compact">
              Batch-ranked designs appear here after filtering. Generated structures are selectable above.
            </div>
          )}
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

function WorkspacePanel({
  project,
  designRuns,
  jobs,
  projectId,
}: {
  project: Project | null;
  designRuns: ProjectDesignRun[];
  jobs: Job[];
  projectId: string;
}) {
  const recentRuns = designRuns.slice(0, 5);
  const activeRun = recentRuns.find((run) => ["preparing", "queued", "running"].includes(run.status));
  const sequenceCount = designRuns.reduce((total, run) => total + (run.generated_sequences?.length ?? 0), 0);
  const structureCount = designRuns.reduce((total, run) => total + (run.generated_structure_ids?.length ?? 0), 0);
  const projectStructureCount = project?.structures?.length ?? 0;

  return (
    <section className="design-panel">
      <div className="section-heading">
        <Folder size={18} />
        <h2>Project workspace</h2>
      </div>
      <div className="design-summary">
        <div>
          <span className="muted">Generation folders</span>
          <strong>{designRuns.length}</strong>
        </div>
        <div>
          <span className="muted">Sequences</span>
          <strong>{sequenceCount}</strong>
        </div>
        <div>
          <span className="muted">Structures</span>
          <strong>{structureCount || projectStructureCount}</strong>
        </div>
      </div>
      {activeRun && (
        <div className="design-active">
          <Loader2 size={17} className="spin" />
          <span>
            {activeRun.library} is {activeRun.status.replace(/_/g, " ")}
          </span>
        </div>
      )}
      <div className="workspace-folders">
        <WorkspaceFolder
          title="Generations"
          count={designRuns.length}
          detail={`${sequenceCount} sequence design(s), ${structureCount} generated structure file(s)`}
        />
        <WorkspaceFolder
          title="Uploaded structures"
          count={projectStructureCount}
          detail={project?.target_original_name ? `Target: ${project.target_original_name}` : "No target uploaded"}
        />
        <WorkspaceFolder
          title="Filtered batches"
          count={jobs.length}
          detail={jobs[0] ? `Latest: ${jobs[0].status}` : "No candidate screens yet"}
        />
      </div>
      <div className="design-run-list">
        {recentRuns.map((run) => (
          <div className="design-run-item" key={run.id}>
            <div>
              <strong>{run.library}</strong>
              <span>{run.prompt}</span>
            </div>
            <div className="design-run-meta">
              <span className={`run-status ${run.status}`}>{run.status.replace(/_/g, " ")}</span>
              <span>{run.generated_sequences?.length ?? 0} seq</span>
              <span>{run.generated_structure_ids?.length ?? 0} pdb</span>
            </div>
            {!!run.generated_sequences?.length && (
              <div className="sequence-preview">
                {run.generated_sequences.slice(0, 3).map((sequence) => (
                  <code key={`${run.id}-${sequence.id}`}>{sequence.sequence.slice(0, 72)}</code>
                ))}
              </div>
            )}
            <div className="design-run-actions">
              <a href={`${API_BASE}/api/projects/${projectId}/design-runs/${run.id}/archive.zip`}>
                <ArrowDownToLine size={15} />
                <span>Outputs</span>
              </a>
              {!!run.generated_sequences?.length && (
                <a href={`${API_BASE}/api/projects/${projectId}/design-runs/${run.id}/sequences.fasta`}>
                  <ArrowDownToLine size={15} />
                  <span>FASTA</span>
                </a>
              )}
            </div>
            {run.error && <p>{run.error}</p>}
          </div>
        ))}
        {!recentRuns.length && (
          <div className="empty-state compact">
            Ask chat to run ProteinMPNN, then generated sequences and structures appear here as project folders.
          </div>
        )}
      </div>
      <Link className="secondary-button design-chat-link" to={`/projects/${projectId}/chat`}>
        <MessageSquare size={17} />
        <span>Open design chat</span>
      </Link>
    </section>
  );
}

function WorkspaceFolder({ title, count, detail }: { title: string; count: number; detail: string }) {
  return (
    <div className="workspace-folder">
      <FileArchive size={17} />
      <div>
        <strong>
          {title} <span>{count}</span>
        </strong>
        <p>{detail}</p>
      </div>
    </div>
  );
}

function ProjectStructureLibrary({
  structures,
  selectedId,
  onSelect,
}: {
  structures: StructureResult[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}) {
  return (
    <section className="structure-library">
      <div className="section-heading">
        <Microscope size={18} />
        <h2>Project structures</h2>
      </div>
      <div className="structure-library-list">
        {structures.map((structure) => (
          <div
            key={structure.id}
            className={structure.id === selectedId ? "structure-library-item active" : "structure-library-item"}
          >
            <button type="button" className="structure-select-button" onClick={() => onSelect(structure.id)}>
              <span>{structure.id === "target" ? "Target" : structure.profile.replace(/_/g, " ")}</span>
              <strong>{structure.pdb_id}</strong>
              <small>{structure.filename}</small>
            </button>
            <a className="structure-download" href={`${API_BASE}${structure.structure_url}`} title="Download structure">
              <ArrowDownToLine size={16} />
            </a>
          </div>
        ))}
        {!structures.length && <div className="empty-state compact">No project structures yet.</div>}
      </div>
    </section>
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
      <ReportPanel markdown={reportMarkdown} onRegionSelect={onRegionSelect} />
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
