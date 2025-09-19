"""
Enhanced AI Chatbot Engine for ForeTel.AI
Using Hugging Face Transformers with local models
"""

import os
import json
import random
import re
from datetime import datetime
from typing import List, Dict, Any
import streamlit as st

# Use lightweight rule-based chatbot to avoid disk space issues
TRANSFORMERS_AVAILABLE = False

class EnhancedChatbot:
    def __init__(self):
        self.conversation_history = []
        self.context = {}
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        
        # Initialize model if transformers available
        if TRANSFORMERS_AVAILABLE:
            self._initialize_model()
        
        # Fallback knowledge base
        self.knowledge_base = {
            "churn_prediction": {
                "keywords": ["churn", "retention", "leave", "cancel", "attrition", "customer loss"],
                "responses": [
                    "Churn prediction helps identify customers at risk of leaving. Our model analyzes usage patterns, payment history, and engagement metrics.",
                    "Customer churn is predicted using machine learning algorithms that consider factors like call duration, data usage, and service complaints.",
                    "Our churn model achieves high accuracy by analyzing customer behavior patterns and identifying early warning signs."
                ]
            },
            "revenue_forecasting": {
                "keywords": ["revenue", "forecast", "prediction", "income", "earnings", "profit"],
                "responses": [
                    "Revenue forecasting uses ensemble models combining multiple algorithms for accurate predictions.",
                    "Our forecasting considers seasonal trends, customer growth, and market conditions to predict future revenue.",
                    "The revenue model analyzes historical data and current trends to provide reliable financial projections."
                ]
            },
            "data_analytics": {
                "keywords": ["data", "analytics", "insights", "analysis", "metrics", "dashboard"],
                "responses": [
                    "Our analytics dashboard provides comprehensive insights into customer behavior and business performance.",
                    "Data analytics helps identify patterns and trends that drive business decisions and strategy.",
                    "We offer real-time analytics with interactive visualizations for better data understanding."
                ]
            },
            "telecom_industry": {
                "keywords": ["telecom", "telecommunications", "mobile", "network", "carrier", "operator"],
                "responses": [
                    "The telecom industry faces unique challenges in customer retention and revenue optimization.",
                    "Telecommunications companies benefit from predictive analytics to improve service quality and customer satisfaction.",
                    "Our platform specializes in telecom-specific metrics and industry best practices."
                ]
            },
            "technical_support": {
                "keywords": ["help", "support", "how to", "tutorial", "guide", "problem", "issue"],
                "responses": [
                    "I'm here to help! You can navigate through different sections using the sidebar menu.",
                    "For technical support, try the documentation or contact our support team through the help section.",
                    "Need assistance? I can guide you through the features and explain how to use the analytics tools."
                ]
            }
        }
        
        # Conversation starters
        self.greetings = [
            "Hello! I'm your ForeTel.AI assistant. How can I help you with telecom analytics today?",
            "Welcome to ForeTel.AI! I'm here to assist with churn prediction and revenue forecasting.",
            "Hi there! Ready to explore some telecom insights? What would you like to know?",
            "Greetings! I'm your AI assistant for telecom analytics. What can I help you discover?"
        ]
        
        # Context-aware responses
        self.contextual_responses = {
            "model_performance": "Our models achieve excellent performance with regular retraining on fresh data.",
            "data_quality": "Data quality is crucial for accurate predictions. We implement comprehensive validation checks.",
            "business_impact": "These analytics directly impact customer retention rates and revenue growth.",
            "implementation": "The platform is designed for easy integration with existing telecom systems."
        }

    def _initialize_model(self):
        """Initialize the Hugging Face model for conversational AI"""
        try:
            # Use a lightweight conversational model
            model_name = "microsoft/DialoGPT-small"  # Smaller model for faster inference
            
            # Check if CUDA is available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Set pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Create pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if device == "cuda" else -1,
                do_sample=True,
                temperature=0.7,
                max_length=150,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            st.success("🤖 Advanced AI chatbot loaded successfully!")
            
        except Exception as e:
            st.warning(f"Could not load advanced model: {e}. Using rule-based responses.")
            self.model = None
            self.pipeline = None

    def preprocess_input(self, user_input: str) -> str:
        """Clean and preprocess user input"""
        # Convert to lowercase and strip whitespace
        processed = user_input.lower().strip()
        
        # Remove special characters but keep spaces and basic punctuation
        processed = re.sub(r'[^\w\s\?\!\.]', '', processed)
        
        return processed

    def extract_intent(self, user_input: str) -> str:
        """Extract user intent from input"""
        processed_input = self.preprocess_input(user_input)
        
        # Check for greetings
        if any(word in processed_input for word in ["hello", "hi", "hey", "greetings"]):
            return "greeting"
        
        # Check knowledge base for intent
        for intent, data in self.knowledge_base.items():
            if any(keyword in processed_input for keyword in data["keywords"]):
                return intent
        
        return "general"

    def generate_ai_response(self, user_input: str) -> str:
        """Generate response using AI model"""
        if not self.pipeline:
            return self.generate_rule_based_response(user_input)
        
        try:
            # Prepare conversation context
            conversation_context = ""
            if self.conversation_history:
                # Include last few exchanges for context
                recent_history = self.conversation_history[-4:]
                for exchange in recent_history:
                    conversation_context += f"Human: {exchange['user']}\nAI: {exchange['bot']}\n"
            
            # Add current input
            full_prompt = f"{conversation_context}Human: {user_input}\nAI:"
            
            # Generate response
            response = self.pipeline(
                full_prompt,
                max_length=len(full_prompt.split()) + 50,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Extract generated text
            generated_text = response[0]['generated_text']
            ai_response = generated_text.split("AI:")[-1].strip()
            
            # Clean up response
            ai_response = ai_response.split("Human:")[0].strip()
            
            # Fallback if response is too short or generic
            if len(ai_response) < 10 or ai_response.lower() in ["yes", "no", "ok", "sure"]:
                return self.generate_rule_based_response(user_input)
            
            return ai_response
            
        except Exception as e:
            st.error(f"AI generation error: {e}")
            return self.generate_rule_based_response(user_input)

    def generate_rule_based_response(self, user_input: str) -> str:
        """Generate response using rule-based system"""
        intent = self.extract_intent(user_input)
        
        if intent == "greeting":
            return random.choice(self.greetings)
        
        elif intent in self.knowledge_base:
            responses = self.knowledge_base[intent]["responses"]
            base_response = random.choice(responses)
            
            # Add contextual information
            if intent == "churn_prediction":
                base_response += " Would you like to see the churn prediction dashboard?"
            elif intent == "revenue_forecasting":
                base_response += " Check out our revenue forecasting section for detailed predictions."
            elif intent == "data_analytics":
                base_response += " Visit the analytics dashboard to explore your data insights."
            
            return base_response
        
        else:
            # General responses with helpful suggestions
            general_responses = [
                "That's an interesting question! I specialize in telecom analytics. Try asking about churn prediction, revenue forecasting, or data insights.",
                "I'm here to help with ForeTel.AI features. You can explore churn analysis, revenue predictions, or customer analytics.",
                "Great question! I can assist with understanding our analytics platform. What specific area interests you most?",
                "I'd be happy to help! Our platform offers comprehensive telecom analytics. What would you like to explore?"
            ]
            
            return random.choice(general_responses)

    def add_context(self, key: str, value: Any):
        """Add context information for better responses"""
        self.context[key] = value

    def get_response(self, user_input: str) -> str:
        """Main method to get chatbot response"""
        if not user_input.strip():
            return "I'm here to help! What would you like to know about ForeTel.AI?"
        
        # Generate response
        if TRANSFORMERS_AVAILABLE and self.pipeline:
            response = self.generate_ai_response(user_input)
        else:
            response = self.generate_rule_based_response(user_input)
        
        # Store conversation
        self.conversation_history.append({
            "user": user_input,
            "bot": response,
            "timestamp": datetime.now().isoformat()
        })
        
        # Limit conversation history
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
        
        return response

    def get_suggestions(self) -> List[str]:
        """Get conversation suggestions"""
        suggestions = [
            "How does churn prediction work?",
            "Show me revenue forecasting insights",
            "What data analytics are available?",
            "Explain the model performance",
            "How can I improve customer retention?",
            "What are the key business metrics?",
            "Tell me about telecom industry trends",
            "How accurate are the predictions?"
        ]
        
        return random.sample(suggestions, 4)

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        self.context = {}

    def export_conversation(self) -> str:
        """Export conversation history as JSON"""
        return json.dumps(self.conversation_history, indent=2)

# Singleton instance
_chatbot_instance = None

def get_chatbot():
    """Get or create chatbot instance"""
    global _chatbot_instance
    if _chatbot_instance is None:
        _chatbot_instance = EnhancedChatbot()
    return _chatbot_instance

# Streamlit Chat Interface Components
def render_chat_message(message: str, is_user: bool = False):
    """Render a chat message with styling"""
    message_class = "user" if is_user else "bot"
    avatar = "🧑‍💼" if is_user else "🤖"
    
    st.markdown(f"""
    <div class="chat-message {message_class}">
        <div style="display: flex; align-items: center; gap: 0.5rem;">
            <span style="font-size: 1.2rem;">{avatar}</span>
            <div>{message}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_typing_indicator():
    """Render typing indicator animation"""
    st.markdown("""
    <div class="chat-message bot">
        <div style="display: flex; align-items: center; gap: 0.5rem;">
            <span style="font-size: 1.2rem;">🤖</span>
            <div class="typing-indicator">
                <span></span><span></span><span></span>
            </div>
        </div>
    </div>
    <style>
    .typing-indicator {
        display: flex;
        gap: 2px;
    }
    .typing-indicator span {
        width: 6px;
        height: 6px;
        background: var(--accent-color);
        border-radius: 50%;
        animation: typing 1.4s infinite ease-in-out;
    }
    .typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
    .typing-indicator span:nth-child(3) { animation-delay: 0.4s; }
    @keyframes typing {
        0%, 60%, 100% { transform: translateY(0); }
        30% { transform: translateY(-10px); }
    }
    </style>
    """, unsafe_allow_html=True)

def render_suggestion_chips(suggestions: List[str], key_prefix: str = ""):
    """Render suggestion chips"""
    st.markdown("💡 **Suggestions:**")
    cols = st.columns(2)
    
    for i, suggestion in enumerate(suggestions):
        col = cols[i % 2]
        if col.button(suggestion, key=f"{key_prefix}_suggestion_{i}", use_container_width=True):
            return suggestion
    
    return None
