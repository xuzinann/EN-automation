from pydantic import BaseModel, Field
import uuid
import time


class DataPoint(BaseModel):
    category: str
    description: str
    value: str | None = None
    confidence: str = "unknown"
    source: str = ""


class Entity(BaseModel):
    name: str
    role: str
    relevance: str


class ProjectContext(BaseModel):
    company_name: str
    industry: str
    deal_type: str
    study_type: str = ""
    num_calls: int = 1
    call_targets: list[str] = []
    key_hypotheses: list[str] = []
    known_data_points: list[DataPoint] = []
    gaps_to_fill: list[str] = []
    key_entities: list[Entity] = []
    expert_profile: str = ""


class Question(BaseModel):
    text: str
    rationale: str = ""
    priority: str = "should_ask"
    target_data: list[str] = []


class GuideSection(BaseModel):
    topic: str
    questions: list[Question] = []
    coverage_threshold: str = ""


class CallGuide(BaseModel):
    opening_script: str = ""
    opening_questions: list[Question] = []
    deep_dive_sections: list[GuideSection] = []
    red_flag_probes: list[Question] = []
    data_points_to_confirm: list[str] = []
    closing_questions: list[Question] = []


class TranscriptEntry(BaseModel):
    timestamp: float = Field(default_factory=time.time)
    speaker: str
    text: str


class StructuredNote(BaseModel):
    timestamp: float = Field(default_factory=time.time)
    category: str
    content: str
    confidence: str = "stated"
    source_quote: str | None = None


class CoverageStatus(BaseModel):
    section: str
    covered_pct: float = 0.0
    answered_questions: list[str] = []
    remaining_questions: list[str] = []


class AgentAction(BaseModel):
    agent_name: str
    action_type: str
    flag_type: str = "should_ask"  # contradiction | must_ask | probe | should_ask | nice_to_have
    content: str
    priority: float = 0.5  # retained for future same-tier tiebreak
    metadata: dict = {}


class CallSession(BaseModel):
    session_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: str = "pre_call"
    project_context: ProjectContext | None = None
    call_guide: CallGuide | None = None
    transcript: list[TranscriptEntry] = []
    notes: list[StructuredNote] = []
    coverage: list[CoverageStatus] = []
    key_takeaways: list[str] = []
    pending_actions: list[AgentAction] = []
    final_summary: str | None = None
