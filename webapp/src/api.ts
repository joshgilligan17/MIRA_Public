export const API_BASE =
  import.meta.env.VITE_MIRA_API_BASE ??
  (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1"
    ? "http://localhost:8000"
    : "");

export type Profile = {
  name: string;
  label: string;
  description: string;
  default_rank_by: string;
  tools: string[];
};

export type SynthesisStatus = {
  configured: boolean;
  provider?: string | null;
  model?: string | null;
};

export type Job = {
  id: string;
  status: "queued" | "running" | "completed" | "failed";
  project_id?: string | null;
  created_at?: string;
  updated_at?: string;
  completed_count: number;
  total_count: number;
  failed_count: number;
  error?: string | null;
  input_files?: string[];
  config: {
    query: string;
    profile: string;
    rank_by: string;
  };
};

export type RankingRow = {
  rank: number;
  pdb_id: string;
  score: number;
};

export type ResidueFeature = {
  kind: string;
  chain?: string | null;
  residue_number?: number | string | null;
  residue_name?: string | null;
  label: string;
  score?: number | null;
};

export type StructureResult = {
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

export type Results = {
  summary: Record<string, number | string | null>;
  ranking: RankingRow[];
  structures: StructureResult[];
  report_synthesis?: {
    mode?: string;
    provider?: string | null;
    model?: string | null;
    error?: string | null;
  };
};

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  created_at: string;
  selected_job_id?: string | null;
  selected_structure_id?: string | null;
  tool_events?: ToolEvent[];
};

export type ToolEvent = {
  tool: string;
  purpose?: string;
  success: boolean;
  data?: string;
  error?: string | null;
  raw?: Record<string, any>;
};

export type ProjectAnalysis = {
  id: string;
  kind: string;
  query: string;
  status: string;
  created_at: string;
  updated_at: string;
  selected_job_id?: string | null;
  selected_structure_id?: string | null;
  tool_events: ToolEvent[];
  metrics: Record<string, any>;
  features: Record<string, any[]>;
  summary: string;
};

export type ProjectDesignRun = {
  id: string;
  library: string;
  prompt: string;
  status: string;
  created_at: string;
  updated_at: string;
  target_structure_id?: string | null;
  output_dir?: string | null;
  command?: string | null;
  generated_structure_ids: string[];
  error?: string | null;
};

export type Project = {
  id: string;
  name: string;
  description: string;
  created_at: string;
  updated_at: string;
  target_file?: string | null;
  target_original_name?: string | null;
  target_uploaded_at?: string | null;
  target_structure?: StructureResult | null;
  structures?: StructureResult[];
  job_ids: string[];
  analysis_ids?: string[];
  design_run_ids?: string[];
  analyses?: ProjectAnalysis[];
  design_runs?: ProjectDesignRun[];
  job_count: number;
  chat_messages: ChatMessage[];
  selected_job_id?: string | null;
  selected_structure_id?: string | null;
};

export type FocusedRegion = {
  evidenceKey: string;
  chain?: string | null;
  residueNumber?: number | string | null;
};

export type BatchJobPayload = {
  files: File[];
  query: string;
  profile: string;
  rankBy: string;
  globPattern?: string;
  chainA?: string;
  chainB?: string;
  maxWorkers: number;
  enableLlmSynthesis: boolean;
  llmProvider?: string;
  llmModel?: string;
  llmBaseUrl?: string;
  llmApiKey?: string;
  llmTemperature?: number;
};

export const rankOptions = [
  { value: "stability", label: "Stability" },
  { value: "buried_surface_area", label: "Buried surface" },
  { value: "n_interface_residues", label: "Interface residues" },
  { value: "mean_bfactor", label: "Mean B-factor" },
  { value: "n_buried", label: "Buried count" },
  { value: "n_exposed", label: "Exposed count" },
  { value: "interface_energy", label: "Interface energy" },
];

export const providerOptions = [
  { value: "", label: "Auto", model: "", baseUrl: "" },
  { value: "openai", label: "OpenAI", model: "gpt-4o-mini", baseUrl: "https://api.openai.com/v1" },
  { value: "minimax", label: "MiniMax", model: "MiniMax-M2.7", baseUrl: "https://api.minimax.io/v1" },
  { value: "anthropic", label: "Anthropic", model: "claude-3-5-haiku-20241022", baseUrl: "https://api.anthropic.com" },
  { value: "azure", label: "Azure", model: "", baseUrl: "" },
];

export async function getHealth(): Promise<{ status: string; synthesis: SynthesisStatus | null }> {
  return apiJson("/api/health");
}

export async function getProfiles(): Promise<Profile[]> {
  const data = await apiJson<{ profiles: Profile[] }>("/api/profiles");
  return data.profiles ?? [];
}

export async function listProjects(): Promise<Project[]> {
  const data = await apiJson<{ projects: Project[] }>("/api/projects");
  return data.projects ?? [];
}

export async function createProject(name: string, description = ""): Promise<Project> {
  const data = await apiJson<{ project: Project }>("/api/projects", {
    method: "POST",
    body: JSON.stringify({ name, description }),
  });
  return data.project;
}

export async function getProject(projectId: string): Promise<Project> {
  const data = await apiJson<{ project: Project }>(`/api/projects/${projectId}`);
  return data.project;
}

export async function updateProject(projectId: string, updates: Partial<Project>): Promise<Project> {
  const data = await apiJson<{ project: Project }>(`/api/projects/${projectId}`, {
    method: "PATCH",
    body: JSON.stringify(updates),
  });
  return data.project;
}

export async function uploadProjectTarget(projectId: string, file: File): Promise<Project> {
  const formData = new FormData();
  formData.append("file", file);
  const data = await apiJson<{ project: Project }>(`/api/projects/${projectId}/target`, {
    method: "POST",
    body: formData,
  });
  return data.project;
}

export async function uploadProjectStructure(
  projectId: string,
  file: File,
): Promise<{ project: Project; structure: StructureResult }> {
  const formData = new FormData();
  formData.append("file", file);
  return apiJson(`/api/projects/${projectId}/structures`, {
    method: "POST",
    body: formData,
  });
}

export async function listProjectJobs(projectId: string): Promise<Job[]> {
  const data = await apiJson<{ jobs: Job[] }>(`/api/projects/${projectId}/jobs`);
  return data.jobs ?? [];
}

export async function createProjectJob(projectId: string, payload: BatchJobPayload): Promise<{ job_id: string; job: Job }> {
  return apiJson(`/api/projects/${projectId}/jobs`, {
    method: "POST",
    body: batchPayloadToFormData(payload),
  });
}

export async function getProjectChat(projectId: string): Promise<ChatMessage[]> {
  const data = await apiJson<{ messages: ChatMessage[] }>(`/api/projects/${projectId}/chat`);
  return data.messages ?? [];
}

export async function sendProjectChat(
  projectId: string,
  message: string,
  selectedJobId?: string | null,
  selectedStructureId?: string | null,
): Promise<{ messages: ChatMessage[]; project?: Project }> {
  const data = await apiJson<{ messages: ChatMessage[]; project?: Project }>(`/api/projects/${projectId}/chat`, {
    method: "POST",
    body: JSON.stringify({
      message,
      selected_job_id: selectedJobId || null,
      selected_structure_id: selectedStructureId || null,
    }),
  });
  return { messages: data.messages ?? [], project: data.project };
}

export async function getJob(jobId: string): Promise<{ job: Job; progress: number }> {
  return apiJson(`/api/jobs/${jobId}`);
}

export async function getResults(jobId: string): Promise<Results> {
  return apiJson(`/api/jobs/${jobId}/results`);
}

export async function getReport(jobId: string): Promise<string> {
  const response = await fetch(`${API_BASE}/api/jobs/${jobId}/report.md`, { credentials: "include" });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.text();
}

async function apiJson<T>(path: string, init: RequestInit = {}): Promise<T> {
  const isForm = init.body instanceof FormData;
  const response = await fetch(`${API_BASE}${path}`, {
    credentials: "include",
    ...init,
    headers: isForm
      ? init.headers
      : {
          "Content-Type": "application/json",
          ...(init.headers ?? {}),
        },
  });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(body?.detail ?? `${response.status} ${response.statusText}`);
  }
  return response.json();
}

function batchPayloadToFormData(payload: BatchJobPayload) {
  const formData = new FormData();
  payload.files.forEach((file) => formData.append("files", file));
  formData.append("query", payload.query);
  formData.append("profile", payload.profile);
  formData.append("rank_by", payload.rankBy);
  formData.append("glob_pattern", payload.globPattern || "*");
  formData.append("chain_a", payload.chainA || "");
  formData.append("chain_b", payload.chainB || "");
  formData.append("max_workers", String(payload.maxWorkers));
  formData.append("enable_llm_synthesis", String(payload.enableLlmSynthesis));
  formData.append("llm_provider", payload.llmProvider || "");
  formData.append("llm_model", payload.llmModel || "");
  formData.append("llm_base_url", payload.llmBaseUrl || "");
  formData.append("llm_api_key", payload.llmApiKey || "");
  formData.append("llm_temperature", String(payload.llmTemperature ?? 0.2));
  return formData;
}
