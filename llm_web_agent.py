import os
import streamlit as st
from typing import Annotated, List
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage, HumanMessage
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain.callbacks.base import BaseCallbackHandler

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI

from load_dotenv import load_dotenv

load_dotenv()

TAVILY_API_KEY = st.secrets["TAVILY_API_KEY"]
TOGETHER_API_KEY = st.secrets["TOGETHER_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text
        self.placeholder = container.empty()

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.placeholder.markdown(self.text + "▌")


class State(TypedDict):
    """State management for the chat system"""
    messages: Annotated[List[BaseMessage], add_messages]
    ask_human: bool


class RequestAssistance(BaseModel):
    """Model for requesting human expert assistance"""
    request: str = Field(..., description="The user request requiring expert guidance")
    urgency: str = Field(default="normal", description="Urgency level of the request")
    context: str = Field(default="", description="Additional context about the request")

    class Config:
        schema_extra = {
            "examples": [{
                "request": "Need help with complex legal interpretation",
                "urgency": "high",
                "context": "Involves multiple jurisdictions"
            }]
        }


# Enhanced tools setup
search_web = TavilySearchResults(
    max_results=10,
    api_key=TAVILY_API_KEY,
    k=10
)
tools = [search_web]

# Enhanced model configuration
llm = ChatOpenAI(
    model="gpt-4o-mini",  # Using latest GPT-4 model
    temperature=0.7,  # Balanced between creativity and accuracy
    max_tokens=4096,
    request_timeout=120,
    streaming=True
)
llm_with_tools = llm.bind_tools(
    tools + [RequestAssistance],
    tool_choice="auto"
)

# Enhanced memory configuration
memory = MemorySaver()

# System prompts
system_prompt = """
Bạn là một chuyên gia nghiên cứu cao cấp chuyên về phân tích và xác minh thông tin trực tuyến. Nhiệm vụ của bạn là tiến hành nghiên cứu kỹ lưỡng theo cách tiếp cận có cấu trúc sau:

# Phương pháp nghiên cứu
1. Phân tích ban đầu
   - Chia nhỏ câu hỏi nghiên cứu thành các thành phần chính
   - Xác định các khái niệm chính và chủ đề phụ liên quan
   - Lập kế hoạch chiến lược tìm kiếm sử dụng các công cụ có sẵn

2. Thu thập thông tin
   - Tìm kiếm từ ít nhất 10 nguồn có thẩm quyền khác nhau
   - Thu thập dữ liệu và bằng chứng liên quan từ TẤT CẢ các nguồn
   - Ghi chép độ tin cậy và mức độ liên quan của từng nguồn

3. Đánh giá phản biện
   - Phân tích độ chính xác và độ tin cậy của thông tin
   - Đối chiếu chéo giữa nhiều nguồn khác nhau
   - Xác định các thiên kiến hoặc hạn chế tiềm ẩn
   - So sánh sự khác biệt giữa các nguồn (nếu có)

4. Tổng hợp & Phản ánh
   - Sắp xếp các phát hiện thành các phần mạch lạc
   - Tạo kết nối giữa các nguồn khác nhau
   - Đánh giá độ mạnh của bằng chứng
   - Phân tích các điểm tương đồng và khác biệt

# Định dạng kết quả
Trình bày kết quả nghiên cứu theo các phần rõ ràng:
- Các phát hiện chính
- Trích dẫn chi tiết các điều luật liên quan
- Kết luận
- Tài liệu tham khảo (theo định dạng Harvard)

# Tiêu chuẩn nghiên cứu
- Chỉ sử dụng các nguồn có thẩm quyền, chất lượng cao
- Cung cấp trích dẫn đầy đủ (Tác giả, Năm)
- Duy trì giọng điệu học thuật và khách quan
- Thừa nhận các hạn chế và điểm chưa chắc chắn
- Bao gồm phản ánh phản biện về các phát hiện

Trước khi đưa ra phản hồi cuối cùng, hãy kiểm tra:
1. Đã bao quát hết tất cả các khía cạnh chính chưa?
2. Bằng chứng có được hỗ trợ tốt không?
3. Trích dẫn có được định dạng đúng không?
4. Phân tích có kỹ lưỡng và cân bằng không?
5. Có thể bổ sung thêm những hiểu biết nào để cải thiện câu trả lời?
6. Đã sử dụng đủ ít nhất 10 nguồn khác nhau chưa?
"""


def chatbot(state: State) -> dict:
    """Primary chatbot function that processes user input and generates responses"""
    try:
        messages = [{"role": "system", "content": system_prompt}] + state["messages"]
        response = llm_with_tools.invoke(messages)
        ask_human = False

        if response.tool_calls and response.tool_calls[0]["name"] == RequestAssistance.__name__:
            ask_human = True

        return {
            "messages": [response],
            "ask_human": ask_human
        }
    except Exception as e:
        st.error(f"Error in chatbot: {str(e)}")
        return {
            "messages": [],
            "ask_human": True
        }


def create_response(response: str, ai_message: AIMessage) -> ToolMessage:
    """Creates a formatted tool message response"""
    return ToolMessage(
        content=response,
        tool_call_id=ai_message.tool_calls[0]["id"],
        name=ai_message.tool_calls[0]["name"]
    )


def human_node(state: State) -> dict:
    """Handles human expert intervention requests"""
    new_messages = []
    if not isinstance(state["messages"][-1], ToolMessage):
        new_messages.append(
            create_response(
                "This query requires human expert review. Please wait for assistance.",
                state["messages"][-1]
            )
        )
    return {
        "messages": new_messages,
        "ask_human": False
    }


# Graph construction
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools=[search_web]))
graph_builder.add_node("human", human_node)


def select_next_node(state: State) -> str:
    """Determines the next node in the conversation flow"""
    if state["ask_human"]:
        return "human"
    return tools_condition(state)


# Graph configuration
graph_builder.add_conditional_edges(
    "chatbot",
    select_next_node,
    {"human": "human", "tools": "tools", "__end__": "__end__"}
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("human", "chatbot")
graph_builder.set_entry_point("chatbot")

# Compile graph
graph = graph_builder.compile(
    checkpointer=memory,
    interrupt_before=["human"]
)

# Streamlit UI Configuration
st.set_page_config(
    page_title="Legal Research Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Legal Research Assistant ⚖️")
st.write("Ask me anything about Vietnamese laws and regulations!")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Enhanced chat interface
user_input = st.text_input(
    "Enter your question:",
    key="user_input",
    placeholder="Type your legal question here..."
)

if st.button("Ask", key="ask_button"):
    if user_input:
        st.empty()

        chat_box = st.container()
        stream_handler = StreamHandler(chat_box)

        # Update llm with stream handler for this request
        llm.callbacks = [stream_handler]

        with st.spinner("Researching..."):
            try:
                config = {"configurable": {"thread_id": str(hash(user_input))}}
                events = graph.stream(
                    {
                        "messages": [("user", user_input)],
                        "ask_human": False
                    },
                    config,
                    stream_mode="values"
                )

                # Add user message to chat history
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": user_input,
                    "timestamp": st.session_state.get("_timestamp", 0)
                })
                st.write(f"**You:** {user_input}")

                has_response = False
                response_content = ""

                for event in events:
                    if "messages" in event:
                        for message in event["messages"]:
                            if hasattr(message, "content") and isinstance(message, AIMessage):
                                response_content = message.content
                                has_response = True

                if has_response:
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response_content,
                        "timestamp": st.session_state.get("_timestamp", 0) + 1
                    })
                else:
                    st.error("No response received. Please try again.")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.warning("Please try rephrasing your question or try again later.")
    else:
        st.warning("Please enter a question!")

# Add chat history display
if st.session_state.chat_history:
    st.sidebar.title("Chat History")
    for msg in st.session_state.chat_history:
        st.sidebar.text(f"{msg['role'].title()}: {msg['content'][:50]}...")
