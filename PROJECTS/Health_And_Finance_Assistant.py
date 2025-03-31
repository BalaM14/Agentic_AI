import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import tools_condition, ToolNode
from typing import Annotated, TypedDict
from io import BytesIO
from IPython.display import Image

# Set OpenAI API Key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Streamlit App Configuration
st.set_page_config(page_title="BMI & Diet Planner", layout="wide")
st.title("💪 Smart BMI, Diet & Budget Planner")
st.write("Enter your details to get personalized health insights! 🚀")
user_message = st.text_area("Enter the Query..!!!")

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

    # Diet Planner
    def generate_diet_plan(bmi_category: str, dietary_preference: str) -> str:
        """
        Generate a diet plan based on BMI category and dietary preference.

        Args:
        bmi_category (str): The BMI classification (Underweight, Normal weight, Overweight, Obese).
        dietary_preference (str): The user's diet type (breakfast, lunch, snack, dinner).

        Returns:
        str: Suggested diet plan.
        """
        diet_plans = {
            "Underweight": {
                "breakfast": "Oatmeal with nuts & banana + Whole milk",
                "lunch": "Grilled chicken/fish with rice & vegetables",
                "snack": "Protein smoothie with nuts & peanut butter",
                "dinner": "Eggs, avocado toast & fruit juice"
            },
            "Normal weight": {
                "breakfast": "Scrambled eggs with whole-grain toast & fruit",
                "lunch": "Grilled salmon with quinoa & greens",
                "snack": "Greek yogurt with honey & berries",
                "dinner": "Chicken stir-fry with vegetables & brown rice"
            },
            "Overweight": {
                "breakfast": "Boiled eggs with whole wheat toast & green tea",
                "lunch": "Grilled chicken with salad & olive oil dressing",
                "snack": "Nuts & seeds mix or hummus with veggies",
                "dinner": "Grilled fish with steamed vegetables"
            },
            "Obese": {
                "breakfast": "Omelette with spinach & black coffee",
                "lunch": "Grilled lean protein (chicken/fish) with steamed broccoli",
                "snack": "Almonds & walnuts",
                "dinner": "Soup with mixed greens & grilled protein"
            }
        }

        meal_plan = diet_plans.get(bmi_category, "No specific plan available")

        # Modify based on dietary preference
        if dietary_preference == "vegetarian":
            for meal in meal_plan:
                meal_plan[meal] = meal_plan[meal].replace("chicken", "paneer").replace("fish", "tofu").replace("eggs", "chickpeas")
        elif dietary_preference == "high_protein":
            for meal in meal_plan:
                meal_plan[meal] += " + Extra protein source (Whey protein, Lentils, Beans)"
        elif dietary_preference == "keto":
            for meal in meal_plan:
                meal_plan[meal] = meal_plan[meal].replace("rice", "cauliflower rice").replace("quinoa", "zucchini noodles").replace("fruit", "avocado")

        return f"Here is your meal plan based on {bmi_category}: {meal_plan}"

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
        return f"Estimated monthly expense for a {diet_type} diet with {'a gym membership' if gym_membership else 'no gym membership'}: ₹{total_cost}"

    # LangChain Agent Setup
    tools = [calculate_bmi, generate_diet_plan, calculate_fitness_expenses]
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)

    class MessagesState(TypedDict):
        """
        Defines the state for message exchange in LangGraph.
        
        Attributes:
        - messages (list): Stores messages exchanged between user and AI.
        """
        messages: Annotated[list, add_messages]

    sys_msg = SystemMessage(content="You are a health assistant providing BMI, diet plans, and financial guidance.")

    def assistant(state: MessagesState):
        """
        Handles user queries and invokes the LangChain AI model.

        Args:
        - state (MessagesState): The current message state.

        Returns:
        - dict: The AI's response messages.
        """
        return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

    builder = StateGraph(MessagesState)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")

    react_graph = builder.compile()

    # Display in Streamlit
    png_image = Image(react_graph.get_graph().draw_mermaid_png())
    image_bytes = BytesIO(png_image.data)
    st.image(image_bytes, caption="Generated Graph", width=300)  # Adjust width as needed
    
    messages = [HumanMessage(content=user_message)]
    messages = react_graph.invoke({"messages": messages})
    
    # Display the Human Message
    st.markdown("### 🗣 **User Query**")
    st.write(f"💬 {messages['messages'][0].content}")

    # Iterate over messages to display tool responses
    st.markdown("---")
    for m in messages["messages"]:
        if isinstance(m, HumanMessage):
            continue  # Skip user message as it's already displayed
        elif isinstance(m, SystemMessage):
            st.markdown("### 🤖 **AI Response**")
            st.write(m.content)
        elif hasattr(m, "name") and m.name:  # Tool message
            st.markdown(f"### 🛠 **{m.name}()**")
            st.write(m.content)
            st.markdown("---")

    # Final AI Summary Response
    st.markdown("### ✅ **Final Insights**")
    st.write(messages["messages"][-1].content)