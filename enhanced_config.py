"""
Enhanced Configuration for ForeTel.AI with Modern UI/UX
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Enhanced UI CSS with Modern Animations and Gestures
ENHANCED_UI_CSS = """
<style>
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global Variables */
:root {
    --primary-color: #667eea;
    --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    --secondary-color: #f093fb;
    --accent-color: #4facfe;
    --success-color: #00d4aa;
    --warning-color: #feca57;
    --error-color: #ff6b6b;
    --dark-bg: #0f0f23;
    --card-bg: rgba(255, 255, 255, 0.05);
    --glass-bg: rgba(255, 255, 255, 0.1);
    --text-primary: #ffffff;
    --text-secondary: #b8bcc8;
    --border-color: rgba(255, 255, 255, 0.1);
    --shadow-light: 0 8px 32px rgba(31, 38, 135, 0.37);
    --shadow-heavy: 0 15px 35px rgba(31, 38, 135, 0.5);
    --border-radius: 16px;
    --transition-fast: 0.2s ease;
    --transition-smooth: 0.4s cubic-bezier(0.4, 0, 0.2, 1);
}

/* Global Styles */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Main App Container */
.stApp {
    background: var(--dark-bg);
    background-image: 
        radial-gradient(circle at 20% 80%, rgba(120, 119, 198, 0.3) 0%, transparent 50%),
        radial-gradient(circle at 80% 20%, rgba(255, 119, 198, 0.3) 0%, transparent 50%),
        radial-gradient(circle at 40% 40%, rgba(120, 200, 255, 0.2) 0%, transparent 50%);
    min-height: 100vh;
    animation: backgroundShift 20s ease-in-out infinite;
}

@keyframes backgroundShift {
    0%, 100% { filter: hue-rotate(0deg); }
    50% { filter: hue-rotate(30deg); }
}

/* Header Styling */
.main-header {
    background: var(--glass-bg);
    backdrop-filter: blur(20px);
    border: 1px solid var(--border-color);
    border-radius: var(--border-radius);
    padding: 2rem;
    margin-bottom: 2rem;
    box-shadow: var(--shadow-light);
    animation: slideInDown 0.8s ease-out;
}

@keyframes slideInDown {
    from {
        transform: translateY(-100px);
        opacity: 0;
    }
    to {
        transform: translateY(0);
        opacity: 1;
    }
}

/* Card Components */
.metric-card, .info-card, .feature-card {
    background: var(--glass-bg);
    backdrop-filter: blur(20px);
    border: 1px solid var(--border-color);
    border-radius: var(--border-radius);
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: var(--shadow-light);
    transition: all var(--transition-smooth);
    cursor: pointer;
    position: relative;
    overflow: hidden;
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
    transition: left 0.6s;
}

.metric-card:hover::before {
    left: 100%;
}

.metric-card:hover, .info-card:hover, .feature-card:hover {
    transform: translateY(-8px) scale(1.02);
    box-shadow: var(--shadow-heavy);
    border-color: var(--accent-color);
}

/* Button Enhancements */
.stButton > button {
    background: var(--primary-gradient) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--border-radius) !important;
    padding: 0.75rem 2rem !important;
    font-weight: 500 !important;
    font-size: 1rem !important;
    transition: all var(--transition-smooth) !important;
    box-shadow: var(--shadow-light) !important;
    position: relative !important;
    overflow: hidden !important;
}

.stButton > button::before {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    width: 0;
    height: 0;
    background: rgba(255,255,255,0.2);
    border-radius: 50%;
    transition: all 0.6s;
    transform: translate(-50%, -50%);
}

.stButton > button:hover::before {
    width: 300px;
    height: 300px;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: var(--shadow-heavy) !important;
}

.stButton > button:active {
    transform: translateY(0) scale(0.98) !important;
}

/* Sidebar Enhancements */
.css-1d391kg {
    background: var(--glass-bg) !important;
    backdrop-filter: blur(20px) !important;
    border-right: 1px solid var(--border-color) !important;
}

/* Navigation Menu */
.nav-item {
    padding: 1rem;
    margin: 0.5rem 0;
    border-radius: var(--border-radius);
    transition: all var(--transition-smooth);
    cursor: pointer;
    border: 1px solid transparent;
}

.nav-item:hover {
    background: var(--glass-bg);
    border-color: var(--accent-color);
    transform: translateX(10px);
}

.nav-item.active {
    background: var(--primary-gradient);
    box-shadow: var(--shadow-light);
}

/* Input Field Enhancements */
.stTextInput > div > div > input,
.stSelectbox > div > div > select,
.stNumberInput > div > div > input {
    background: var(--glass-bg) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: var(--border-radius) !important;
    color: var(--text-primary) !important;
    backdrop-filter: blur(10px) !important;
    transition: all var(--transition-smooth) !important;
}

.stTextInput > div > div > input:focus,
.stSelectbox > div > div > select:focus,
.stNumberInput > div > div > input:focus {
    border-color: var(--accent-color) !important;
    box-shadow: 0 0 0 2px rgba(79, 172, 254, 0.2) !important;
    transform: scale(1.02) !important;
}

/* Chart Container */
.chart-container {
    background: var(--glass-bg);
    backdrop-filter: blur(20px);
    border: 1px solid var(--border-color);
    border-radius: var(--border-radius);
    padding: 2rem;
    margin: 1rem 0;
    box-shadow: var(--shadow-light);
    animation: fadeInUp 0.8s ease-out;
}

@keyframes fadeInUp {
    from {
        transform: translateY(30px);
        opacity: 0;
    }
    to {
        transform: translateY(0);
        opacity: 1;
    }
}

/* Loading Animation */
.loading-spinner {
    display: inline-block;
    width: 40px;
    height: 40px;
    border: 3px solid var(--border-color);
    border-radius: 50%;
    border-top-color: var(--accent-color);
    animation: spin 1s ease-in-out infinite;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

/* Success/Error Messages */
.success-message {
    background: linear-gradient(135deg, var(--success-color), #00b894);
    color: white;
    padding: 1rem 1.5rem;
    border-radius: var(--border-radius);
    margin: 1rem 0;
    animation: slideInRight 0.5s ease-out;
}

.error-message {
    background: linear-gradient(135deg, var(--error-color), #e17055);
    color: white;
    padding: 1rem 1.5rem;
    border-radius: var(--border-radius);
    margin: 1rem 0;
    animation: shake 0.5s ease-in-out;
}

@keyframes slideInRight {
    from {
        transform: translateX(100px);
        opacity: 0;
    }
    to {
        transform: translateX(0);
        opacity: 1;
    }
}

@keyframes shake {
    0%, 100% { transform: translateX(0); }
    25% { transform: translateX(-5px); }
    75% { transform: translateX(5px); }
}

/* Chat Interface */
.chat-container {
    background: var(--glass-bg);
    backdrop-filter: blur(20px);
    border: 1px solid var(--border-color);
    border-radius: var(--border-radius);
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: var(--shadow-light);
    max-height: 500px;
    overflow-y: auto;
}

.chat-message {
    padding: 1rem;
    margin: 0.5rem 0;
    border-radius: 12px;
    animation: messageSlide 0.3s ease-out;
    position: relative;
}

.chat-message.user {
    background: var(--primary-gradient);
    color: white;
    margin-left: 20%;
    text-align: right;
}

.chat-message.bot {
    background: var(--glass-bg);
    color: var(--text-primary);
    margin-right: 20%;
    border: 1px solid var(--border-color);
}

@keyframes messageSlide {
    from {
        transform: translateY(20px);
        opacity: 0;
    }
    to {
        transform: translateY(0);
        opacity: 1;
    }
}

/* Floating Action Button */
.fab {
    position: fixed;
    bottom: 2rem;
    right: 2rem;
    width: 60px;
    height: 60px;
    background: var(--primary-gradient);
    border-radius: 50%;
    border: none;
    color: white;
    font-size: 1.5rem;
    cursor: pointer;
    box-shadow: var(--shadow-heavy);
    transition: all var(--transition-smooth);
    z-index: 1000;
}

.fab:hover {
    transform: scale(1.1) rotate(15deg);
    box-shadow: 0 20px 40px rgba(31, 38, 135, 0.6);
}

/* Progress Bar */
.progress-bar {
    width: 100%;
    height: 8px;
    background: var(--border-color);
    border-radius: 4px;
    overflow: hidden;
    margin: 1rem 0;
}

.progress-fill {
    height: 100%;
    background: var(--primary-gradient);
    border-radius: 4px;
    transition: width var(--transition-smooth);
    animation: progressGlow 2s ease-in-out infinite;
}

@keyframes progressGlow {
    0%, 100% { box-shadow: 0 0 5px var(--accent-color); }
    50% { box-shadow: 0 0 20px var(--accent-color); }
}

/* Tooltip */
.tooltip {
    position: relative;
    display: inline-block;
}

.tooltip .tooltiptext {
    visibility: hidden;
    width: 200px;
    background: var(--dark-bg);
    color: var(--text-primary);
    text-align: center;
    border-radius: 8px;
    padding: 0.5rem;
    position: absolute;
    z-index: 1;
    bottom: 125%;
    left: 50%;
    margin-left: -100px;
    opacity: 0;
    transition: opacity var(--transition-smooth);
    border: 1px solid var(--border-color);
    box-shadow: var(--shadow-light);
}

.tooltip:hover .tooltiptext {
    visibility: visible;
    opacity: 1;
}

/* Mobile Responsiveness */
@media (max-width: 768px) {
    .main-header {
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    .metric-card, .info-card, .feature-card {
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .chat-message.user {
        margin-left: 10%;
    }
    
    .chat-message.bot {
        margin-right: 10%;
    }
    
    .fab {
        bottom: 1rem;
        right: 1rem;
        width: 50px;
        height: 50px;
        font-size: 1.2rem;
    }
}

/* Accessibility */
@media (prefers-reduced-motion: reduce) {
    * {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }
}

/* Custom Scrollbar */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: var(--border-color);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb {
    background: var(--accent-color);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--primary-color);
}

/* Particle Animation Background */
.particles {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    z-index: -1;
}

.particle {
    position: absolute;
    width: 2px;
    height: 2px;
    background: var(--accent-color);
    border-radius: 50%;
    animation: float 6s ease-in-out infinite;
}

@keyframes float {
    0%, 100% {
        transform: translateY(0) rotate(0deg);
        opacity: 0.7;
    }
    50% {
        transform: translateY(-20px) rotate(180deg);
        opacity: 1;
    }
}

/* Glassmorphism Effects */
.glass-panel {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: var(--border-radius);
    box-shadow: var(--shadow-light);
}

/* Hover Effects for Interactive Elements */
.interactive:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-heavy);
    transition: all var(--transition-smooth);
}

/* Status Indicators */
.status-online {
    color: var(--success-color);
    animation: pulse 2s infinite;
}

.status-offline {
    color: var(--error-color);
}

@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.5; }
    100% { opacity: 1; }
}

/* Data Visualization Enhancements */
.chart-wrapper {
    position: relative;
    overflow: hidden;
    border-radius: var(--border-radius);
}

.chart-wrapper::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(45deg, transparent 30%, rgba(255,255,255,0.05) 50%, transparent 70%);
    pointer-events: none;
    animation: chartShimmer 3s ease-in-out infinite;
}

@keyframes chartShimmer {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
}
</style>
"""

# Enhanced JavaScript for Interactions
ENHANCED_JS = """
<script>
// Enhanced UI Interactions and Animations

// Particle System
function createParticles() {
    const particleContainer = document.createElement('div');
    particleContainer.className = 'particles';
    document.body.appendChild(particleContainer);
    
    for (let i = 0; i < 50; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        particle.style.left = Math.random() * 100 + '%';
        particle.style.top = Math.random() * 100 + '%';
        particle.style.animationDelay = Math.random() * 6 + 's';
        particle.style.animationDuration = (Math.random() * 3 + 3) + 's';
        particleContainer.appendChild(particle);
    }
}

// Smooth Scroll Behavior
function initSmoothScroll() {
    document.documentElement.style.scrollBehavior = 'smooth';
}

// Interactive Card Tilt Effect
function initCardTilt() {
    const cards = document.querySelectorAll('.metric-card, .info-card, .feature-card');
    
    cards.forEach(card => {
        card.addEventListener('mousemove', (e) => {
            const rect = card.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            const centerX = rect.width / 2;
            const centerY = rect.height / 2;
            
            const rotateX = (y - centerY) / 10;
            const rotateY = (centerX - x) / 10;
            
            card.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) translateZ(10px)`;
        });
        
        card.addEventListener('mouseleave', () => {
            card.style.transform = 'perspective(1000px) rotateX(0) rotateY(0) translateZ(0)';
        });
    });
}

// Typing Animation for Text
function typeWriter(element, text, speed = 50) {
    let i = 0;
    element.innerHTML = '';
    
    function type() {
        if (i < text.length) {
            element.innerHTML += text.charAt(i);
            i++;
            setTimeout(type, speed);
        }
    }
    
    type();
}

// Progress Bar Animation
function animateProgress(element, targetWidth, duration = 1000) {
    let start = 0;
    const startTime = performance.now();
    
    function animate(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        const currentWidth = start + (targetWidth - start) * progress;
        element.style.width = currentWidth + '%';
        
        if (progress < 1) {
            requestAnimationFrame(animate);
        }
    }
    
    requestAnimationFrame(animate);
}

// Intersection Observer for Animations
function initScrollAnimations() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.animation = 'fadeInUp 0.8s ease-out forwards';
            }
        });
    }, { threshold: 0.1 });
    
    document.querySelectorAll('.metric-card, .chart-container, .info-card').forEach(el => {
        observer.observe(el);
    });
}

// Enhanced Button Ripple Effect
function initButtonRipples() {
    document.addEventListener('click', function(e) {
        if (e.target.tagName === 'BUTTON') {
            const button = e.target;
            const rect = button.getBoundingClientRect();
            const size = Math.max(rect.width, rect.height);
            const x = e.clientX - rect.left - size / 2;
            const y = e.clientY - rect.top - size / 2;
            
            const ripple = document.createElement('span');
            ripple.style.cssText = `
                position: absolute;
                width: ${size}px;
                height: ${size}px;
                left: ${x}px;
                top: ${y}px;
                background: rgba(255,255,255,0.3);
                border-radius: 50%;
                transform: scale(0);
                animation: ripple 0.6s linear;
                pointer-events: none;
            `;
            
            button.style.position = 'relative';
            button.style.overflow = 'hidden';
            button.appendChild(ripple);
            
            setTimeout(() => ripple.remove(), 600);
        }
    });
}

// Theme Switcher
function initThemeSwitcher() {
    const themeToggle = document.createElement('button');
    themeToggle.innerHTML = '🌙';
    themeToggle.className = 'fab theme-toggle';
    themeToggle.style.bottom = '8rem';
    themeToggle.onclick = toggleTheme;
    document.body.appendChild(themeToggle);
}

function toggleTheme() {
    document.documentElement.classList.toggle('light-theme');
    const toggle = document.querySelector('.theme-toggle');
    toggle.innerHTML = document.documentElement.classList.contains('light-theme') ? '☀️' : '🌙';
}

// Real-time Clock
function initClock() {
    const clockElement = document.createElement('div');
    clockElement.className = 'digital-clock';
    clockElement.style.cssText = `
        position: fixed;
        top: 1rem;
        right: 1rem;
        background: var(--glass-bg);
        backdrop-filter: blur(20px);
        padding: 0.5rem 1rem;
        border-radius: 8px;
        color: var(--text-primary);
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
        border: 1px solid var(--border-color);
        z-index: 1000;
    `;
    
    function updateClock() {
        const now = new Date();
        clockElement.textContent = now.toLocaleTimeString();
    }
    
    updateClock();
    setInterval(updateClock, 1000);
    document.body.appendChild(clockElement);
}

// Initialize all enhancements
document.addEventListener('DOMContentLoaded', function() {
    setTimeout(() => {
        createParticles();
        initSmoothScroll();
        initCardTilt();
        initScrollAnimations();
        initButtonRipples();
        initThemeSwitcher();
        initClock();
    }, 1000);
});

// CSS for ripple animation
const style = document.createElement('style');
style.textContent = `
    @keyframes ripple {
        to {
            transform: scale(4);
            opacity: 0;
        }
    }
    
    .light-theme {
        --dark-bg: #f8fafc;
        --card-bg: rgba(0, 0, 0, 0.05);
        --glass-bg: rgba(0, 0, 0, 0.1);
        --text-primary: #1a202c;
        --text-secondary: #4a5568;
        --border-color: rgba(0, 0, 0, 0.1);
    }
`;
document.head.appendChild(style);
</script>
"""

# Chatbot Configuration
CHATBOT_CONFIG = {
    "model_name": "microsoft/DialoGPT-medium",
    "max_length": 1000,
    "temperature": 0.7,
    "do_sample": True,
    "pad_token_id": 50256,
    "responses": {
        "greeting": [
            "Hello! I'm your ForeTel.AI assistant. How can I help you with telecom analytics today?",
            "Welcome to ForeTel.AI! I'm here to assist with churn prediction and revenue forecasting.",
            "Hi there! Ready to explore some telecom insights? What would you like to know?"
        ],
        "churn": [
            "Churn prediction helps identify customers likely to leave. Would you like to analyze specific customer segments?",
            "Our churn model uses advanced ML to predict customer behavior. What metrics interest you most?",
            "Customer retention is crucial! Let me help you understand the churn patterns in your data."
        ],
        "revenue": [
            "Revenue forecasting uses ensemble models for accurate predictions. What time period interests you?",
            "Our revenue models consider multiple factors like customer behavior and market trends.",
            "Let's dive into revenue analytics! I can help explain the forecasting methodology."
        ],
        "default": [
            "I'm here to help with telecom analytics. Try asking about churn prediction, revenue forecasting, or data insights!",
            "That's an interesting question! Could you be more specific about what telecom metrics you'd like to explore?",
            "I specialize in telecom analytics. Feel free to ask about customer data, predictions, or business insights!"
        ]
    }
}

# Environment Variables
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/foretel_db")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
DEBUG = os.getenv("DEBUG", "True").lower() == "true"

# API Configuration
API_CONFIG = {
    "base_url": "http://localhost:8000",
    "timeout": 30,
    "retry_attempts": 3
}

# Feature Flags
FEATURE_FLAGS = {
    "enhanced_ui": True,
    "chatbot_enabled": True,
    "animations_enabled": True,
    "dark_mode": True,
    "real_time_updates": True,
    "advanced_analytics": True
}
