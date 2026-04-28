/* ═══════════════════════════════════════════════════════════════════
   LearnPath AI — Frontend Application
   Connects to Flask API, orchestrates all views and agent interactions
   ═══════════════════════════════════════════════════════════════════ */

const API = '';  // same origin
let USER_ID = localStorage.getItem('lp_user_id') || generateUID();
let currentQuiz = null;
let allTopics = [];
let currentRoadmap = null;

localStorage.setItem('lp_user_id', USER_ID);

/* ── UTILITIES ──────────────────────────────────────────────────── */

function generateUID() {
  return 'user_' + Math.random().toString(36).slice(2, 10);
}

async function apiFetch(path, options = {}) {
  const res = await fetch(API + path, {
    headers: { 'Content-Type': 'application/json', 'X-User-ID': USER_ID, ...(options.headers || {}) },
    ...options,
  });
  const json = await res.json();
  if (!res.ok && !json.success) throw new Error(json.error || 'API error');
  return json;
}

function toast(msg, type = 'info') {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.className = `toast show ${type}`;
  clearTimeout(el._t);
  el._t = setTimeout(() => el.classList.remove('show'), 3000);
}

function showView(name) {
  document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
  document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
  document.getElementById(`view-${name}`).classList.add('active');
  document.querySelector(`[data-view="${name}"]`)?.classList.add('active');

  if (name === 'dashboard') loadDashboard();
  if (name === 'roadmap') loadRoadmapView();
  if (name === 'topics') loadTopics();
  if (name === 'metrics') loadMetrics();
}

function setLoading(elId, msg = 'Loading...') {
  document.getElementById(elId).innerHTML = `
    <div class="loading-state"><div class="spinner"></div><p>${msg}</p></div>`;
}

function levelTag(level) {
  return `<span class="tag level-${level}">${level}</span>`;
}

function typeIcon(type) {
  const icons = { course: '🎓', video: '▶️', book: '📖', documentation: '📄', article: '📝', paper: '🔬' };
  return icons[type] || '🔗';
}

/* ── NAV ─────────────────────────────────────────────────────────── */

document.querySelectorAll('.nav-link').forEach(link => {
  link.addEventListener('click', e => {
    e.preventDefault();
    showView(link.dataset.view);
  });
});

/* ── ONBOARDING ─────────────────────────────────────────────────── */

document.querySelectorAll('.tag.selectable').forEach(tag => {
  tag.addEventListener('click', () => tag.classList.toggle('selected'));
});

async function submitOnboarding() {
  const name = document.getElementById('onboard-name').value.trim() || 'Learner';
  const goal = document.getElementById('onboard-goal').value.trim();
  const hours = parseInt(document.getElementById('onboard-hours').value);

  if (!goal) { toast('Please enter what you want to learn!', 'error'); return; }

  const selected = [...document.querySelectorAll('.tag.selectable.selected')].map(t => t.dataset.id);

  const btn = document.querySelector('#onboarding-modal .btn-primary');
  btn.textContent = '⚡ Generating your path...';
  btn.disabled = true;

  try {
    // Create user
    await apiFetch('/api/user', {
      method: 'POST',
      body: JSON.stringify({ name, goal, hours_per_week: hours }),
    });

    document.getElementById('user-pill').textContent = name;

    // Generate roadmap
    const res = await apiFetch('/api/roadmap', {
      method: 'POST',
      body: JSON.stringify({ goal, hours_per_week: hours, current_knowledge: selected }),
    });

    currentRoadmap = res.data.roadmap;

    document.getElementById('onboarding-modal').classList.remove('active');
    toast(`✅ ${currentRoadmap.total_steps}-step learning path created!`, 'success');
    showView('roadmap');
  } catch (e) {
    toast(e.message, 'error');
  } finally {
    btn.textContent = 'Generate My Learning Path ⚡';
    btn.disabled = false;
  }
}

/* ── DASHBOARD ───────────────────────────────────────────────────── */

async function loadDashboard() {
  setLoading('dashboard-content', 'Loading your dashboard...');
  try {
    const [dashRes, userRes] = await Promise.all([
      apiFetch('/api/progress'),
      apiFetch('/api/user'),
    ]);
    const d = dashRes.data;
    const user = userRes.data;
    const s = d.summary;

    document.getElementById('user-pill').textContent = user.name || 'Learner';

    // Build MCP context display
    const mcp = await buildMCPDisplay(user, s, d);

    document.getElementById('dashboard-content').innerHTML = `
      <div class="dashboard-grid">

        <div class="stats-row">
          <div class="stat-card">
            <div class="stat-num purple">${s.overall_progress_pct}%</div>
            <div class="stat-label">Overall Progress</div>
            <div class="progress-bar-wrap" style="margin-top:10px">
              <div class="progress-bar" style="width:${s.overall_progress_pct}%"></div>
            </div>
          </div>
          <div class="stat-card">
            <div class="stat-num teal">${s.completed}</div>
            <div class="stat-label">Topics Completed</div>
          </div>
          <div class="stat-card">
            <div class="stat-num amber">${s.in_progress}</div>
            <div class="stat-label">In Progress</div>
          </div>
          <div class="stat-card">
            <div class="stat-num blue">${s.average_quiz_score > 0 ? s.average_quiz_score + '%' : '—'}</div>
            <div class="stat-label">Avg Quiz Score</div>
          </div>
        </div>

        ${mcp}

        ${d.recent_quiz_history.length > 0 ? `
        <div>
          <div class="section-title">Recent Quiz Activity</div>
          <div style="display:flex;flex-direction:column;gap:8px">
            ${d.recent_quiz_history.slice(0,5).map(q => `
              <div class="card card-sm" style="display:flex;align-items:center;justify-content:space-between">
                <div>
                  <span style="font-weight:600">${q.topic_id.replace(/_/g,' ').replace(/\b\w/g,c=>c.toUpperCase())}</span>
                  <span style="color:var(--text3);font-size:12px;margin-left:10px">${new Date(q.taken_at*1000).toLocaleDateString()}</span>
                </div>
                <div style="display:flex;align-items:center;gap:10px">
                  <span style="font-size:13px;color:var(--text2)">${q.correct}/${q.total} correct</span>
                  <span class="tag ${q.passed ? 'level-beginner' : 'level-advanced'}">${q.score_pct}%</span>
                </div>
              </div>
            `).join('')}
          </div>
        </div>
        ` : ''}

        ${s.total_topics === 0 ? `
        <div class="card" style="text-align:center;padding:32px">
          <div style="font-size:40px;margin-bottom:12px">🎯</div>
          <h3 style="margin-bottom:8px">Ready to start learning?</h3>
          <p style="color:var(--text2);margin-bottom:16px">Set a goal to generate your personalized learning roadmap.</p>
          <button class="btn btn-primary" onclick="document.getElementById('onboarding-modal').classList.add('active')">
            Set Learning Goal ⚡
          </button>
        </div>
        ` : `
        <div style="display:flex;gap:10px;flex-wrap:wrap">
          <button class="btn btn-primary" onclick="showView('roadmap')">📍 View Roadmap</button>
          <button class="btn btn-outline" onclick="showView('chat')">💬 Ask AI Tutor</button>
          <button class="btn btn-outline" onclick="document.getElementById('onboarding-modal').classList.add('active')">↺ New Goal</button>
        </div>
        `}

      </div>`;
  } catch (e) {
    document.getElementById('dashboard-content').innerHTML = `<div class="empty-state"><p>Error loading dashboard: ${e.message}</p></div>`;
  }
}

async function buildMCPDisplay(user, summary, dashboard) {
  const currentTopic = dashboard.current_topic || '—';
  return `
    <div class="mcp-panel">
      <h3>🔗 MCP Structured Context <span style="font-size:11px;color:var(--text3);font-weight:400">(sent to LLM on every call)</span></h3>
      <div class="mcp-grid">
        <div class="mcp-field">
          <div class="key">user_goal</div>
          <div class="val">${user.goal || 'Not set'}</div>
        </div>
        <div class="mcp-field">
          <div class="key">current_step</div>
          <div class="val">${currentTopic.replace(/_/g,' ')}</div>
        </div>
        <div class="mcp-field">
          <div class="key">progress_pct</div>
          <div class="val">${summary.overall_progress_pct}%</div>
        </div>
        <div class="mcp-field">
          <div class="key">avg_quiz_score</div>
          <div class="val">${summary.average_quiz_score > 0 ? summary.average_quiz_score + '%' : 'No quizzes yet'}</div>
        </div>
        <div class="mcp-field">
          <div class="key">completed / total</div>
          <div class="val">${summary.completed} / ${summary.total_topics}</div>
        </div>
        <div class="mcp-field">
          <div class="key">retrieved_docs</div>
          <div class="val">[ injected at query time ]</div>
        </div>
      </div>
    </div>`;
}

/* ── ROADMAP VIEW ─────────────────────────────────────────────────── */

async function loadRoadmapView() {
  setLoading('roadmap-content', 'Loading your roadmap...');
  try {
    let roadmap = currentRoadmap;
    if (!roadmap) {
      const res = await apiFetch('/api/roadmap');
      roadmap = res.data.roadmap;
      currentRoadmap = roadmap;
    }

    const progRes = await apiFetch('/api/progress');
    const topicProgress = {};
    progRes.data.topics.forEach(t => { topicProgress[t.topic_id] = t; });

    renderRoadmap(roadmap, topicProgress);
  } catch (e) {
    document.getElementById('roadmap-content').innerHTML = `
      <div class="empty-state">
        <p style="margin-bottom:16px">No roadmap yet. Set a learning goal to get started.</p>
        <button class="btn btn-primary" onclick="document.getElementById('onboarding-modal').classList.add('active')">
          Set Learning Goal ⚡
        </button>
      </div>`;
  }
}

function renderRoadmap(roadmap, topicProgress = {}) {
  const html = `
    <div class="roadmap-container">
      <div class="roadmap-meta">
        <div class="meta-badge">🎯 Goal: <strong>${roadmap.goal}</strong></div>
        <div class="meta-badge">📚 <strong>${roadmap.total_steps}</strong> topics</div>
        <div class="meta-badge">⏱ <strong>${roadmap.total_hours}h</strong> total</div>
        <div class="meta-badge">📅 ~<strong>${roadmap.weeks_to_complete}</strong> weeks</div>
        <div class="meta-badge">🌐 Domain: <strong>${roadmap.domain}</strong></div>
      </div>

      <div class="roadmap-steps">
        ${roadmap.steps.map(step => {
          const prog = topicProgress[step.topic_id] || {};
          const status = prog.status || step.status || 'not_started';
          const statusClass = status === 'completed' ? 'completed' : status === 'in_progress' ? 'in-progress' : '';
          const statusIcon = status === 'completed' ? '✅' : status === 'in_progress' ? '🔄' : step.step;
          const score = prog.score_pct > 0 ? `· Score: ${prog.score_pct}%` : '';

          return `
          <div class="step-card ${statusClass}" id="step-${step.topic_id}">
            <div class="step-num ${statusClass}">${statusIcon}</div>
            <div class="step-body">
              <div class="step-title">${step.title}</div>
              <div class="step-why">${step.why_needed}</div>
              <div class="step-meta">
                ${levelTag(step.level)}
                <span class="tag">⏱ ${step.duration_hours}h</span>
                <span class="tag">📂 ${step.topic}</span>
                ${score ? `<span class="tag level-beginner">${score}</span>` : ''}
              </div>
              <div class="step-actions">
                ${status !== 'in_progress' && status !== 'completed' ? `
                  <button class="btn btn-sm btn-outline" onclick="markStatus('${step.topic_id}', 'in_progress')">▶ Start</button>
                ` : ''}
                ${status === 'in_progress' ? `
                  <button class="btn btn-sm btn-teal" onclick="markStatus('${step.topic_id}', 'completed')">✓ Mark Complete</button>
                ` : ''}
                <button class="btn btn-sm btn-outline" onclick="openResources('${step.topic_id}', '${step.title}')">📚 Resources</button>
                <button class="btn btn-sm btn-outline" onclick="openQuiz('${step.topic_id}', '${step.title}')">📝 Quiz</button>
                <button class="btn btn-sm btn-outline" onclick="askAbout('${step.title}')">💬 Ask AI</button>
              </div>
            </div>
          </div>`;
        }).join('')}
      </div>
    </div>`;

  document.getElementById('roadmap-content').innerHTML = html;
}

async function markStatus(topicId, status) {
  try {
    await apiFetch(`/api/progress/${topicId}`, {
      method: 'PUT',
      body: JSON.stringify({ status }),
    });
    toast(status === 'completed' ? '✅ Marked as complete!' : '▶ Started!', 'success');
    loadRoadmapView();
  } catch (e) {
    toast(e.message, 'error');
  }
}

/* ── CHAT ────────────────────────────────────────────────────────── */

function setChat(msg) {
  document.getElementById('chat-input').value = msg;
  document.getElementById('chat-input').focus();
}

function askAbout(topic) {
  showView('chat');
  setChat(`Explain ${topic} and why it's important for my learning goal.`);
}

function appendMessage(role, content, meta = {}) {
  const container = document.getElementById('chat-messages');
  const div = document.createElement('div');
  div.className = `message ${role}`;

  const formattedContent = content
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\n/g, '<br>');

  let metaHtml = '';
  if (meta.confidence !== undefined) {
    const pct = Math.round(meta.confidence * 100);
    metaHtml += `<span>Confidence: ${pct}% <span class="confidence-bar"><span class="confidence-fill" style="width:${pct}%"></span></span></span>`;
  }
  if (meta.response_time_ms) {
    metaHtml += `<span>${Math.round(meta.response_time_ms)}ms</span>`;
  }

  let docsHtml = '';
  if (meta.retrieved_docs && meta.retrieved_docs.length > 0) {
    docsHtml = `<div class="retrieved-docs">
      <strong>📎 Retrieved from knowledge base:</strong>
      ${meta.retrieved_docs.map(d => `<div>• ${d.topic} (${d.level}) — score: ${d.score}</div>`).join('')}
    </div>`;
  }

  div.innerHTML = `
    <div class="msg-avatar">${role === 'user' ? '🧑' : '🤖'}</div>
    <div class="msg-bubble">
      ${formattedContent}
      ${docsHtml}
      ${metaHtml ? `<div class="msg-meta">${metaHtml}</div>` : ''}
    </div>`;

  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
}

function appendTyping() {
  const container = document.getElementById('chat-messages');
  const div = document.createElement('div');
  div.className = 'message ai';
  div.id = 'typing-indicator';
  div.innerHTML = `
    <div class="msg-avatar">🤖</div>
    <div class="msg-bubble" style="color:var(--text3)">
      <span class="spinner" style="width:16px;height:16px;border-width:2px;display:inline-block;vertical-align:middle;margin-right:8px"></span>
      Thinking...
    </div>`;
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
}

async function sendChat() {
  const input = document.getElementById('chat-input');
  const msg = input.value.trim();
  if (!msg) return;

  input.value = '';
  appendMessage('user', msg);
  appendTyping();

  try {
    const res = await apiFetch('/api/chat', {
      method: 'POST',
      body: JSON.stringify({ message: msg }),
    });
    document.getElementById('typing-indicator')?.remove();
    const d = res.data;
    appendMessage('ai', d.answer, {
      confidence: d.confidence,
      response_time_ms: d.response_time_ms,
      retrieved_docs: d.retrieved_docs,
    });
  } catch (e) {
    document.getElementById('typing-indicator')?.remove();
    appendMessage('ai', '⚠️ Error: ' + e.message);
  }
}

// Initial welcome message
setTimeout(() => {
  if (document.getElementById('chat-messages').children.length === 0) {
    appendMessage('ai', `**Hello! I'm your AI Tutor powered by RAG** 🎓\n\nI answer questions grounded in the learning knowledge base. Ask me about:\n- Concepts in your roadmap (ML, Python, Linear Algebra, etc.)\n- "Why do I need X for my goal?"\n- "How does Y work?"\n\nAll my answers come from retrieved knowledge base documents — no hallucination!`);
  }
}, 500);

/* ── TOPICS VIEW ─────────────────────────────────────────────────── */

async function loadTopics() {
  setLoading('topics-content', 'Loading topics...');
  try {
    const res = await apiFetch('/api/topics');
    allTopics = res.data.topics;
    renderTopics(allTopics);
  } catch (e) {
    document.getElementById('topics-content').innerHTML = `<div class="empty-state">${e.message}</div>`;
  }
}

function filterTopics(query) {
  const q = query.toLowerCase();
  const filtered = allTopics.filter(t =>
    t.id.includes(q) || t.topic.toLowerCase().includes(q) || t.level.includes(q)
  );
  renderTopics(filtered);
}

function renderTopics(topics) {
  // Group by domain
  const groups = {};
  topics.forEach(t => {
    if (!groups[t.topic]) groups[t.topic] = [];
    groups[t.topic].push(t);
  });

  const html = Object.entries(groups).map(([domain, items]) => `
    <div style="margin-bottom:8px">
      <div style="padding:12px 28px 6px;font-size:12px;font-weight:600;color:var(--text3);text-transform:uppercase;letter-spacing:0.6px">${domain}</div>
      <div class="topics-grid" style="padding-top:0">
        ${items.map(t => `
          <div class="topic-card">
            <h3>${t.id.replace(/_/g,' ').replace(/\b\w/g,c=>c.toUpperCase())}</h3>
            <div class="topic-meta">
              ${levelTag(t.level)}
              <span class="tag">⏱ ${t.duration_hours}h</span>
            </div>
            ${t.prerequisites.length > 0 ? `
              <div style="font-size:11px;color:var(--text3);margin-bottom:10px">
                Prereqs: ${t.prerequisites.map(p => p.replace(/_/g,' ')).join(', ')}
              </div>` : '<div style="margin-bottom:10px"></div>'}
            <div class="topic-actions">
              <button class="btn btn-sm btn-outline" onclick="openResources('${t.id}', '${t.id.replace(/_/g,' ').replace(/\b\w/g,c=>c.toUpperCase())}')">📚 Resources</button>
              <button class="btn btn-sm btn-outline" onclick="openQuiz('${t.id}', '${t.id.replace(/_/g,' ').replace(/\b\w/g,c=>c.toUpperCase())}')">📝 Quiz</button>
            </div>
          </div>
        `).join('')}
      </div>
    </div>
  `).join('');

  document.getElementById('topics-content').innerHTML = html || '<div class="empty-state">No topics found.</div>';
}

/* ── QUIZ ────────────────────────────────────────────────────────── */

async function openQuiz(topicId, title) {
  document.getElementById('quiz-title').textContent = `📝 Quiz: ${title}`;
  document.getElementById('quiz-body').innerHTML = `<div class="loading-state"><div class="spinner"></div><p>Loading quiz...</p></div>`;
  document.getElementById('quiz-modal').classList.add('active');

  try {
    const res = await apiFetch(`/api/quiz/${topicId}`);
    currentQuiz = res.data;
    currentQuiz.topic_id = topicId;
    currentQuiz.userAnswers = {};
    renderQuiz();
  } catch (e) {
    document.getElementById('quiz-body').innerHTML = `<div class="empty-state">${e.message}</div>`;
  }
}

function renderQuiz() {
  const q = currentQuiz;
  if (!q.questions || q.questions.length === 0) {
    document.getElementById('quiz-body').innerHTML = `<div class="empty-state">No questions available for this topic yet.</div>`;
    return;
  }

  const html = `
    <div style="padding:20px 24px">
      <div style="margin-bottom:16px;color:var(--text2);font-size:13px">
        ${q.total_questions} questions · ${levelTag(q.level || 'beginner')}
      </div>
      ${q.questions.map(question => `
        <div class="quiz-question" id="qq-${question.id}">
          <p>${question.id + 1}. ${question.question}</p>
          <div class="quiz-options">
            ${question.options.map((opt, i) => `
              <div class="quiz-option" id="opt-${question.id}-${i}"
                onclick="selectOption(${question.id}, ${i})">
                <span style="color:var(--text3);font-weight:600">${String.fromCharCode(65+i)}.</span>
                ${opt}
              </div>
            `).join('')}
          </div>
        </div>
      `).join('')}
      <div style="display:flex;justify-content:flex-end;gap:10px;margin-top:16px">
        <button class="btn btn-outline" onclick="closeQuiz()">Cancel</button>
        <button class="btn btn-primary" onclick="submitQuiz()">Submit Answers →</button>
      </div>
    </div>`;

  document.getElementById('quiz-body').innerHTML = html;
}

function selectOption(questionId, optionIndex) {
  currentQuiz.userAnswers[questionId] = optionIndex;

  // Deselect all options for this question
  document.querySelectorAll(`[id^="opt-${questionId}-"]`).forEach(el => el.classList.remove('selected'));
  document.getElementById(`opt-${questionId}-${optionIndex}`).classList.add('selected');
}

async function submitQuiz() {
  const unanswered = currentQuiz.questions.filter(q => currentQuiz.userAnswers[q.id] === undefined);
  if (unanswered.length > 0) {
    toast(`Please answer all ${unanswered.length} remaining question(s)`, 'error');
    return;
  }

  try {
    const res = await apiFetch(`/api/quiz/${currentQuiz.topic_id}/submit`, {
      method: 'POST',
      body: JSON.stringify({ answers: currentQuiz.userAnswers }),
    });

    const result = res.data;

    // Show correct/incorrect on options
    result.results.forEach(r => {
      const correctEl = document.getElementById(`opt-${r.question_id}-${r.correct_answer}`);
      const userEl = document.getElementById(`opt-${r.question_id}-${r.user_answer}`);
      if (correctEl) correctEl.classList.add('correct');
      if (userEl && !r.is_correct) userEl.classList.add('incorrect');
    });

    // Show result panel
    const resultHtml = `
      <div class="quiz-result">
        <div class="score-circle" style="border-color:${result.passed ? 'var(--teal)' : 'var(--coral)'}">
          <div class="score-num" style="color:${result.passed ? 'var(--teal)' : 'var(--coral)'}">${result.score_pct}%</div>
          <div class="score-label">${result.passed ? 'PASSED' : 'RETRY'}</div>
        </div>
        <h3 style="margin-bottom:8px">${result.correct}/${result.total} correct</h3>
        <p style="color:var(--text2);margin-bottom:20px">${result.feedback}</p>
        <div style="display:flex;gap:10px;justify-content:center;flex-wrap:wrap">
          ${result.passed ? `<button class="btn btn-teal" onclick="markStatus('${currentQuiz.topic_id}','completed');closeQuiz()">✅ Mark Complete</button>` : ''}
          <button class="btn btn-outline" onclick="openQuiz('${currentQuiz.topic_id}','${currentQuiz.title}')">↺ Retry</button>
          <button class="btn btn-outline" onclick="closeQuiz()">Close</button>
        </div>
      </div>`;

    // Append result below questions
    const existingResult = document.getElementById('quiz-result-panel');
    if (existingResult) existingResult.remove();
    const panel = document.createElement('div');
    panel.id = 'quiz-result-panel';
    panel.innerHTML = resultHtml;
    document.getElementById('quiz-body').appendChild(panel);
    panel.scrollIntoView({ behavior: 'smooth' });

    toast(result.passed ? `🎉 Passed with ${result.score_pct}%!` : `Score: ${result.score_pct}% — keep studying!`,
          result.passed ? 'success' : 'info');
  } catch (e) {
    toast(e.message, 'error');
  }
}

function closeQuiz() {
  document.getElementById('quiz-modal').classList.remove('active');
  currentQuiz = null;
}

/* ── RESOURCES MODAL ─────────────────────────────────────────────── */

async function openResources(topicId, title) {
  document.getElementById('resources-title').textContent = `📚 Resources: ${title}`;
  document.getElementById('resources-body').innerHTML = `<div class="loading-state"><div class="spinner"></div></div>`;
  document.getElementById('resources-modal').classList.add('active');

  try {
    const res = await apiFetch(`/api/resources/${topicId}`);
    const d = res.data;
    const resources = d.resources || [];

    const html = `
      <div style="padding:20px 24px">
        <div style="display:flex;gap:14px;margin-bottom:20px;flex-wrap:wrap">
          <div class="card card-sm" style="flex:1;text-align:center">
            <div style="font-size:22px;font-weight:700;color:var(--purple-light)">${d.summary.total_resources}</div>
            <div style="font-size:12px;color:var(--text2)">Total Resources</div>
          </div>
          <div class="card card-sm" style="flex:1;text-align:center">
            <div style="font-size:22px;font-weight:700;color:var(--teal)">${d.summary.free_resources}</div>
            <div style="font-size:12px;color:var(--text2)">Free</div>
          </div>
          <div class="card card-sm" style="flex:1;text-align:center">
            <div style="font-size:22px;font-weight:700;color:var(--amber)">${d.summary.estimated_total_hours}h</div>
            <div style="font-size:12px;color:var(--text2)">Estimated</div>
          </div>
        </div>

        ${resources.length === 0 ? '<div class="empty-state">No resources yet for this topic.</div>' :
          resources.map(r => `
            <div class="resource-item">
              <div class="resource-icon">${typeIcon(r.type)}</div>
              <div class="resource-info">
                <div class="r-title"><a href="${r.url}" target="_blank">${r.title}</a></div>
                <div class="r-meta">
                  ${r.type} · ${r.estimated_hours}h
                  ${r.free ? '<span style="color:var(--teal)">· Free</span>' : '<span style="color:var(--amber)">· Paid</span>'}
                </div>
              </div>
              <a href="${r.url}" target="_blank" class="btn btn-sm btn-outline">Open →</a>
            </div>
          `).join('')}

        ${d.related_topics.length > 0 ? `
          <div style="margin-top:16px">
            <div class="section-title">Related Topics</div>
            <div style="display:flex;gap:8px;flex-wrap:wrap">
              ${d.related_topics.map(t => `
                <button class="chip" onclick="closeResources();openResources('${t.topic_id}','${t.title}')">
                  ${t.title} (${(t.relevance_score*100).toFixed(0)}% match)
                </button>`).join('')}
            </div>
          </div>` : ''}
      </div>`;

    document.getElementById('resources-body').innerHTML = html;
  } catch (e) {
    document.getElementById('resources-body').innerHTML = `<div class="empty-state">${e.message}</div>`;
  }
}

function closeResources() {
  document.getElementById('resources-modal').classList.remove('active');
}

/* ── METRICS VIEW ────────────────────────────────────────────────── */

async function loadMetrics() {
  setLoading('metrics-content', 'Loading observability data...');
  try {
    const res = await apiFetch('/api/metrics');
    const d = res.data;
    const sys = d.system_metrics;
    const vs = d.vector_store;

    const html = `
      <div class="metrics-grid">

        <div class="metrics-row">
          <div class="metric-card">
            <h3>Total API Requests</h3>
            <div class="metric-val">${sys.requests?.total ?? 0}</div>
            <div class="metric-sub">Avg ${(sys.requests?.avg_response_time_ms ?? 0).toFixed(0)}ms response time</div>
          </div>
          <div class="metric-card">
            <h3>RAG Retrievals</h3>
            <div class="metric-val">${sys.rag?.total_retrievals ?? 0}</div>
            <div class="metric-sub">Avg top score: ${(sys.rag?.avg_top_score ?? 0).toFixed(3)}</div>
          </div>
          <div class="metric-card">
            <h3>Agent Calls</h3>
            <div class="metric-val">${sys.agents?.total_calls ?? 0}</div>
            <div class="metric-sub">Success rate: ${(sys.agents?.success_rate_pct ?? 0).toFixed(1)}%</div>
          </div>
        </div>

        <div class="card">
          <div class="section-title" style="margin-bottom:14px">Vector Store Stats</div>
          <div style="display:flex;gap:20px;flex-wrap:wrap">
            <div>
              <div style="font-size:28px;font-weight:700;color:var(--purple-light)">${vs.total_documents}</div>
              <div style="font-size:12px;color:var(--text2)">Document Chunks Indexed</div>
            </div>
            <div>
              <div style="font-size:28px;font-weight:700;color:var(--teal)">${vs.vocabulary_size}</div>
              <div style="font-size:12px;color:var(--text2)">Vocabulary Terms (TF-IDF)</div>
            </div>
          </div>
        </div>

        <div class="card">
          <div class="section-title" style="margin-bottom:14px">Active Agents</div>
          <div style="display:flex;flex-wrap:wrap;gap:6px">
            ${d.agents.active.map(a => `
              <div class="agent-pill">
                <span class="dot green"></span> ${a}
              </div>`).join('')}
          </div>
        </div>

        ${sys.recent_requests && sys.recent_requests.length > 0 ? `
        <div class="card">
          <div class="section-title" style="margin-bottom:14px">Recent Requests</div>
          <table class="log-table">
            <thead>
              <tr><th>Endpoint</th><th>Status</th><th>Response Time</th><th>Timestamp</th></tr>
            </thead>
            <tbody>
              ${sys.recent_requests.map(r => `
                <tr>
                  <td>${r.endpoint}</td>
                  <td><span class="tag ${r.status_code === 200 ? 'level-beginner' : 'level-advanced'}">${r.status_code}</span></td>
                  <td>${r.response_time_ms ? r.response_time_ms.toFixed(0) + 'ms' : '—'}</td>
                  <td>${new Date(r.timestamp * 1000).toLocaleTimeString()}</td>
                </tr>
              `).join('')}
            </tbody>
          </table>
        </div>` : ''}

        <div class="card" style="background:var(--bg3)">
          <div class="section-title" style="margin-bottom:10px">Guardrails Status</div>
          <div style="display:flex;flex-direction:column;gap:8px;font-size:13px">
            <div style="display:flex;align-items:center;gap:10px">
              <span class="dot green"></span>
              <span><strong>Input Guardrails:</strong> Active — blocking harmful content, validating query length</span>
            </div>
            <div style="display:flex;align-items:center;gap:10px">
              <span class="dot green"></span>
              <span><strong>Output Guardrails:</strong> Active — confidence threshold ${(0.05*100).toFixed(0)}%, fallback on low-confidence</span>
            </div>
            <div style="display:flex;align-items:center;gap:10px">
              <span class="dot green"></span>
              <span><strong>RAG Grounding:</strong> Active — answers sourced only from knowledge base</span>
            </div>
            <div style="display:flex;align-items:center;gap:10px">
              <span class="dot green"></span>
              <span><strong>Upload Validation:</strong> Active — PDF/TXT only, 5MB limit</span>
            </div>
          </div>
        </div>

      </div>`;

    document.getElementById('metrics-content').innerHTML = html;
  } catch (e) {
    document.getElementById('metrics-content').innerHTML = `<div class="empty-state">${e.message}</div>`;
  }
}

/* ── KEYBOARD SHORTCUTS ──────────────────────────────────────────── */

document.addEventListener('keydown', e => {
  if (e.key === 'Escape') {
    document.querySelectorAll('.modal-overlay.active').forEach(m => {
      if (m.id !== 'onboarding-modal') m.classList.remove('active');
    });
  }
});

/* ── INIT ────────────────────────────────────────────────────────── */

(async function init() {
  try {
    // Check if user already has data
    const userRes = await apiFetch('/api/user');
    const user = userRes.data;

    if (user.goal) {
      // Returning user — skip onboarding
      document.getElementById('onboarding-modal').classList.remove('active');
      document.getElementById('user-pill').textContent = user.name || 'Learner';
      document.getElementById('onboard-name').value = user.name || 'Learner';
      document.getElementById('onboard-goal').value = user.goal || '';
      loadDashboard();
    }
    // else: modal stays open for new user
  } catch (e) {
    // Fresh user — onboarding modal stays
    console.log('New user — showing onboarding');
  }
})();
