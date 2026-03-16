// chat.js - handles sending messages, rendering responses, health checks

const chatMessages = document.getElementById('chatMessages');
const chatForm = document.getElementById('chatForm');
const questionInput = document.getElementById('questionInput');
const sendBtn = document.getElementById('sendBtn');
const sendIcon = document.getElementById('sendIcon');
const sendSpinner = document.getElementById('sendSpinner');
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const welcomeCard = document.getElementById('welcomeCard');

const API_URL = '/api/chat';
let isLoading = false;
let messagesContainer = null;

function getMessagesContainer() {
    if (!messagesContainer) {
        messagesContainer = chatMessages.querySelector('.space-y-3, .space-y-4') || chatMessages;
    }
    return messagesContainer;
}

// auto-resize textarea
questionInput.addEventListener('input', () => {
    questionInput.style.height = 'auto';
    questionInput.style.height = Math.min(questionInput.scrollHeight, 120) + 'px';
});

// form submit
chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const q = questionInput.value.trim();
    if (!q || isLoading) return;
    await sendMessage(q);
});

// enter to send, shift+enter for newline
questionInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        chatForm.dispatchEvent(new Event('submit'));
    }
});

async function sendMessage(question) {
    // hide welcome screen
    if (welcomeCard && welcomeCard.parentNode) {
        welcomeCard.style.opacity = '0';
        welcomeCard.style.transform = 'scale(0.97)';
        welcomeCard.style.transition = 'all 0.25s ease';
        setTimeout(() => welcomeCard.remove(), 250);
    }

    appendMessage('user', question);
    questionInput.value = '';
    questionInput.style.height = 'auto';

    setLoading(true);
    const loader = appendLoading();

    try {
        const res = await fetch(API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question }),
        });

        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || `Server error (${res.status})`);
        }

        const data = await res.json();
        loader.remove();
        appendMessage('assistant', data.answer, data.sources || []);

    } catch (err) {
        loader.remove();
        appendMessage(
            'assistant',
            `⚠️ Could not get a response: ${err.message}.\n\nMake sure the backend is running:\n\`python backend/main.py\``,
            []
        );
    } finally {
        setLoading(false);
    }
}

function appendMessage(role, content, sources = []) {
    const container = getMessagesContainer();

    const wrapper = document.createElement('div');
    wrapper.className = `flex ${role === 'user' ? 'justify-end' : 'justify-start'} ${role === 'user' ? 'animate-slide-right' : 'animate-slide-left'}`;

    const bubble = document.createElement('div');
    bubble.className = `px-4 py-3 ${role === 'user' ? 'msg-user' : 'msg-assistant'}`;

    if (role === 'user') {
        bubble.innerHTML = `<p class="text-sm text-gray-200 whitespace-pre-wrap">${escapeHtml(content)}</p>`;
    } else {
        const html = formatMarkdown(content);
        bubble.innerHTML = `
            <div class="assistant-content text-sm text-gray-300 leading-relaxed">${html}</div>
            ${sources.length > 0 ? `
                <div class="mt-2.5 pt-2 border-t border-white/5 flex flex-wrap gap-1.5">
                    <span class="text-[10px] text-gray-500 mr-0.5 self-center">Sources:</span>
                    ${sources.map(s => `<span class="source-tag">📄 ${escapeHtml(s)}</span>`).join('')}
                </div>` : ''}
        `;
    }

    wrapper.appendChild(bubble);
    container.appendChild(wrapper);
    scrollToBottom();
}

function appendLoading() {
    const container = getMessagesContainer();
    const el = document.createElement('div');
    el.className = 'flex justify-start animate-slide-left';
    el.innerHTML = `
        <div class="msg-assistant px-4 py-3.5 flex items-center gap-1.5">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        </div>`;
    container.appendChild(el);
    scrollToBottom();
    return el;
}

function setLoading(on) {
    isLoading = on;
    sendBtn.disabled = on;
    sendIcon.classList.toggle('hidden', on);
    sendSpinner.classList.toggle('hidden', !on);
    questionInput.disabled = on;
    if (!on) questionInput.focus();
}

function askSuggestion(btn) {
    const span = btn.querySelector('span:last-child');
    const q = span ? span.textContent.trim() : '';
    if (q && !isLoading) {
        questionInput.value = q;
        sendMessage(q);
    }
}

// helpers
function escapeHtml(text) {
    const d = document.createElement('div');
    d.textContent = text;
    return d.innerHTML;
}

function formatMarkdown(text) {
    let html = escapeHtml(text);
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');
    html = html.replace(/`([^`]+)`/g, '<code class="bg-white/10 px-1 py-0.5 rounded text-xs text-accent-400">$1</code>');
    html = html.replace(/---/g, '<hr>');
    html = html.split('\n\n').map(p => p.trim()).filter(Boolean)
        .map(p => `<p>${p.replace(/\n/g, '<br>')}</p>`).join('');
    return html;
}

function scrollToBottom() {
    requestAnimationFrame(() => {
        chatMessages.scrollTo({ top: chatMessages.scrollHeight, behavior: 'smooth' });
    });
}

// health check
async function checkHealth() {
    try {
        const res = await fetch('/api/health');
        const data = await res.json();
        if (data.pipeline_ready) {
            statusDot.className = 'w-2 h-2 rounded-full bg-green-400 animate-pulse';
            statusText.textContent = 'Online';
        } else {
            statusDot.className = 'w-2 h-2 rounded-full bg-yellow-400 animate-pulse';
            statusText.textContent = 'Loading…';
        }
    } catch {
        statusDot.className = 'w-2 h-2 rounded-full bg-red-400';
        statusText.textContent = 'Offline';
    }
}

checkHealth();
setInterval(checkHealth, 15000);
window.addEventListener('load', () => questionInput.focus());

// handle mobile keyboard pushing layout
if ('visualViewport' in window) {
    window.visualViewport.addEventListener('resize', () => {
        document.getElementById('app').style.height = window.visualViewport.height + 'px';
    });
}
