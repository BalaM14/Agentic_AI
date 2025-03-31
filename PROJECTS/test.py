import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import tools_condition, ToolNode
from typing import Annotated, TypedDict
import os

# Set OpenAI API Key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Streamlit App Configuration
st.set_page_config(page_title="BMI & Diet Planner", layout="wide")

# Title
st.title("💪 BMI, Diet Plan & Expense Calculator")
st.write("Enter your details to get personalized health insights! 🚀")

# Sidebar for User Input
st.sidebar.header("User Information")
age = st.sidebar.number_input("Age", min_value=10, max_value=100, value=30)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
weight_kg = st.sidebar.number_input("Weight (kg)", min_value=20.0, max_value=200.0, value=75.0)
height_m = st.sidebar.number_input("Height (m)", min_value=1.0, max_value=2.5, value=1.75)

# Additional Preferences
diet_preference = st.sidebar.selectbox("Dietary Preference", ["Balanced", "Vegetarian", "High Protein", "Keto"])
gym_membership = st.sidebar.checkbox("Do you have a gym membership?", value=True)

# Submit Button
if st.sidebar.button("Generate Insights"):
    # BMI Calculation Tool
    def calculate_bmi(weight_kg: float, height_m: float, age: int, gender: str) -> str:
        """
        Calculate BMI based on weight and height.

        Args:
        weight_kg (float): Weight in kilograms.
        height_m (float): Height in meters.
        age (int): Age of the person.
        gender (str): Gender of the person.

        Returns:
        str: BMI value and category with insights.
        """
        bmi = weight_kg / (height_m ** 2)
        category = (
            "Underweight" if bmi < 18.5 else
            "Normal weight" if bmi < 24.9 else
            "Overweight" if bmi < 29.9 else
            "Obese"
        )
        insight = ""
        if age < 18:
            insight = "For children/teens, use BMI percentiles instead of fixed ranges."
        elif age >= 65:
            insight = "For older adults, BMI may not reflect body fat accurately due to muscle loss."
        if gender.lower() == "female" and bmi > 24.9:
            insight += " Women naturally have a higher body fat percentage, so BMI alone may not be the best measure."
        return f"BMI: {round(bmi, 2)}, Category: {category}. {insight}"

    # Diet Plan Generation Tool
    def generate_diet_plan(bmi_category: str, dietary_preference: str) -> str:
        """
        Generate a diet plan based on BMI category and dietary preference.

        Args:
        bmi_category (str): The BMI classification (Underweight, Normal, Overweight, Obese).
        dietary_preference (str): The user's diet type (Balanced, Vegetarian, High Protein, Keto).

        Returns:
        str: Suggested diet plan.
        """
        diet_plans = {
            "Underweight": {
                "Balanced": "Include calorie-dense foods like nuts, dairy, and protein-rich meals.",
                "Vegetarian": "Add high-calorie plant-based foods like avocados, nuts, and legumes.",
                "High Protein": "Consume lean meats, eggs, dairy, and protein shakes.",
                "Keto": "Eat high-fat foods like cheese, nuts, and avocados."
            },
            "Normal weight": {
                "Balanced": "A mix of lean proteins, whole grains, and healthy fats.",
                "Vegetarian": "Focus on plant proteins, whole grains, and fiber-rich foods.",
                "High Protein": "Maintain muscle mass with lean meat, fish, and eggs.",
                "Keto": "Moderate protein intake with high-fat and low-carb foods."
            },
            "Overweight": {
                "Balanced": "Reduce sugar and refined carbs, increase fiber intake.",
                "Vegetarian": "Limit starchy vegetables, focus on legumes and healthy fats.",
                "High Protein": "Low-carb, high-protein meals with lean meats, fish, and eggs.",
                "Keto": "Strict low-carb, high-fat diet to promote fat loss."
            },
            "Obese": {
                "Balanced": "Follow a strict calorie-deficit diet with whole foods.",
                "Vegetarian": "Focus on high-fiber foods like beans and non-starchy vegetables.",
                "High Protein": "Emphasize lean protein sources like chicken, turkey, and fish.",
                "Keto": "Very low-carb, high-fat diet for rapid weight loss."
            }
        }
        return f"Diet Plan ({bmi_category}, {dietary_preference}): {diet_plans.get(bmi_category, {}).get(dietary_preference, 'No specific plan available')}."

    # Expense Calculation Tool
    def calculate_fitness_expenses(diet_type: str, gym_membership: bool) -> str:
        """
        Calculate estimated monthly expenses for fitness based on diet and gym membership.

        Args:
        diet_type (str): The user's diet type (Balanced, Vegetarian, High Protein, Keto).
        gym_membership (bool): Whether the user has a gym membership.

        Returns:
        str: Estimated monthly expense.
        """
        cost_estimates = {
            "Balanced": 5000,
            "Vegetarian": 4500,
            "High Protein": 6000,
            "Keto": 7000
        }
        gym_cost = 2000 if gym_membership else 0
        total_cost = cost_estimates.get(diet_type, 5000) + gym_cost
        return f"Estimated monthly expense: ₹{total_cost} ({diet_type} diet + {'Gym' if gym_membership else 'No Gym'})"

    # LangChain Agent Setup
    tools = [calculate_bmi, generate_diet_plan, calculate_fitness_expenses]
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)

    class MessagesState(TypedDict):
        messages: Annotated[list, add_messages]

    sys_msg = SystemMessage(content="You are a health assistant providing BMI, diet plans, and financial guidance.")

    def assistant(state: MessagesState):
        return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

    builder = StateGraph(MessagesState)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")

    react_graph = builder.compile()
    user_message = f"I am {age} years old, {gender.lower()}, {weight_kg} kg, and {height_m} meters tall. Calculate my BMI and suggest a diet plan for {diet_preference} diet with expenses."

    messages = [HumanMessage(content=user_message)]
    messages = react_graph.invoke({"messages": messages})

    

    # Display tool names at the end
    st.subheader("🔧 **Tools Used:**")
    for tool in tools:
        st.write(f"- {tool.__name__}")

    # Display Results in Streamlit
    st.subheader("📝 Results")

    print(messages['messages'])

    for m in messages['messages']:
        if m.content is not None:
            st.write(m.content)
            st.write("============================================")