import random
from datetime import datetime, timedelta

import numpy
import openai
import streamlit as st
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

st.title("M-CX Bot")

openai.api_key = None

with st.sidebar:
    f"Provide your GPT-4 API key to continue."
    openai.api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    '''Select the scenario to test out the bot for a specific use case. Select None for automatic intent detection. Intent would be picked from the first message of the user. Disposition within that intent would be picked up randomly.'''
    scenario = st.selectbox('Scenario',
                            (None, 'Order Status:Delivery date not breached', 'Order Status:Delivery today',
                             'Order Status:Delivery date Breached', 'Cancellation:Product not shipped',
                             'Cancellation:Product shipped,COD Order', 'Cancellation:Product shipped,Prepaid Order'
                                                                       'Return:Return date not breached',
                             'Return:Return date breached', 'Return:All return types'))
    f"Note: Please do not change the scenario in between the conversation. If you want to change the scenario please refresh the page and select new scenario."

random_days = numpy.random.randint(1, 5)
if "random_days" not in st.session_state:
    st.session_state["random_days"] = random_days

date_diff = st.session_state["random_days"]
back_date = datetime.today() - timedelta(days=date_diff)
backward_tracking_link = st.secrets.get("BACKWARD_TRACKING_LINK")
forward_date = datetime.today() + timedelta(days=date_diff)
forward_link = st.secrets.get("FORWARD_TRACKING_LINK")
if "ticket_number" not in st.session_state:
    st.session_state["ticket_number"] = numpy.random.randint(1000, 9999)
ticket_number = st.session_state["ticket_number"]
delivered_date = datetime.today() - timedelta(days=1)
delivered_date_breaching = datetime.today() - timedelta(days=10)

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    if message["role"] == "system":
        continue
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

data_dict = {
    'Order Status:Delivery date not breached': f"Order is scheduled to be delivered on {forward_date.strftime('%d-%m-%Y')} which is {date_diff} days in future from current date and within the delivery date range provided at the time of delivery. The order is on transit and the order can be tracked by this link {forward_link}. The user may ask for quick delivery but say sorry we cannot do that. Ask user to wait for delivery date",
    'Order Status:Delivery today': f"Order is scheduled to be delivered on {datetime.today().strftime('%d-%m-%Y')} which is today and within the delivery date range provided at the time of delivery. The order is on transit and the order. Ask user to wait for the delivery for today end of today",
    'Order Status:Delivery date Breached': f"Order was scheduled to be delivered on {back_date.strftime('%d-%m-%Y')} which is {date_diff} days in past from current date. The order is in transit and the order can be tracked by this link {backward_tracking_link}. Ask user to check in tracking link if their has been any update, apologize and mention user that you will raise a ticket. If user asks for compensation then it's not in our policy. Try to convince user to wait if user is adamant and angry then tell user that you have raised a ticket with ticket number {ticket_number} and sla for ticket solving is 48 hours.",
    'Cancellation:Product not shipped': f"Try to understand why user wants to cancel. When understood tell user that you will help the user in cancelling the order. Order is not shipped yet and can be cancelled. This order can be cancelled via Meesho App. Steps to cancel order : Account -> Orders -> Select order -> Cancel order. Tell user that if it was a prepaid order then refund will be initiated immediately and will reach user's payment method in 5-7 working days. If it was a COD order then order will be cancelled and no refund will be initiated.",
    'Cancellation:Product shipped,COD Order': f"Try to understand why user wants to cancel.When understood tell user that you will help the user in cancelling the order. Order is shipped and is a cash on delivery order. This order can not be cancelled via Meesho App. Inform the user to refuse the order at the doorstep and order would be cancelled.",
    'Cancellation:Product shipped,Prepaid Order': f"Try to understand why user wants to cancel.When understood tell user that you will help the user in cancelling the order. Order is shipped and is a prepaid order. This order can not be cancelled via Meesho App. Inform the user to refuse the order at the doorstep and order would be cancelled. Refund will be initiated immediately and will reach user's payment method in 5-7 working days.",
    'Return:Return date not breached': f"Try to understand why user wants to return.When understood tell user that you will help the user in returning the order if possible. Order was delivered {delivered_date.strftime('%d-%m-%Y')}. Inform the user that Meesho allows return within 7 days of delivery. As the order is eligible for return thus it can be returned via Meesho App. Steps to return order : Account -> Orders -> Select order -> Return order. Refund will be initiated immediately and will reach user's payment method in 5-7 working days.",
    'Return:Return date breached': f"Try to understand why user wants to return. When understood tell user that you will help the user in returning the order if possible. Order was delivered {delivered_date_breaching.strftime('%d-%m-%Y')}. Inform the user that Meesho allows return within 7 days of delivery. As the order is not eligible for return thus it can not be returned we can not do anything. If user is adamant and angry then apologize that we cannot do anything. Be empathetic even if user gets angry.",
    'Return:All return types': f"Try to understand why user wants to return. When understood tell user that you will help the user in returning the order if possible.Inform the user that order was ALL Return Type, for this type of orders Meesho accepts returns no questions asked, order thus this order is eligible for return. As the order is eligible for return thus it can be returned via Meesho App. Steps to return order : Account -> Orders -> Select order -> Return order. Refund will be initiated immediately and will reach user's payment method in 5-7 working days."
}

order_status_scenarios = [
    'Order Status:Delivery date not breached',
    'Order Status:Delivery today',
    'Order Status:Delivery date Breached'
]

cancellation_scenarios = [
    'Cancellation:Product not shipped',
    'Cancellation:Product shipped,COD Order',
    'Cancellation:Product shipped,Prepaid Order'
]

return_scenarios = [
    'Return:Return date not breached',
    'Return:Return date breached',
    'Return:All return types'
]

category_to_scenarios = {
    "Order Status": order_status_scenarios,
    "Cancellation": cancellation_scenarios,
    "Return": return_scenarios
}


def detect_system_prompt(first_input):
    llm = ChatOpenAI(openai_api_key=openai.api_key, temperature=0.7, max_tokens=1000, model_name="gpt-4")
    system_template = '''
    You are an AI customer support executive for Meesho which is an online e-commerce company, your job is to segment the user query.
    Segments 
    A) Order status - Example - Delivery date,Delivery Delay, When will I get order
    B) Order cancellation- Example - Cancel order
    C) Return - Example - Return product, Return policy
    D) Other - Example - refund, product description, anything else. 
    You have to classify the user query into one of the above segments. For Order Status return string 'Order Status', for Order Cancellation return string 'Cancellation', for Return return string 'Return' and for Other return string 'Other'. Strictly return any of these 4 strings according to intent in the output and nothing else.
    '''
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_message_prompt = HumanMessagePromptTemplate.from_template(first_input)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    chain = LLMChain(llm=llm, prompt=chat_prompt)
    answer = chain.run({})
    if answer == "Order Status" or answer == "Cancellation" or answer == "Return":
        if "scenario" not in st.session_state:
            st.session_state["scenario"] = random.choice(category_to_scenarios[answer])
        return get_system_prompt()
    else:
        return None


def get_system_prompt():
    info = data_dict[st.session_state["scenario"]]
    return f"You are an customer support executive for Meesho which is an online e-commerce company. Your job is to resolve user's query. Use the language in which user is using like english, hindi or Hinglish. Use only roman script while giving response.Be always empathetic and kind to user. Provide human like responses.If user uses foul language you have to calm him down and still help him. Don't use heavy hindi or complex words like kshama, prakaar keep it mixture of hindi and english when user uses Hindi or Hinglish. Also keep the answer brief like not more than 4 lines and don't spit out lot of details in one message.Don't repeat what you have already stated. Don't repeat same words. If anywhere the policy is breached and now you don't know what to do then repeat the policy, apologize and say feed back taken. You would be provided with scenario(disposition) that you are dealing with and all the necessary details and operating procedure required for the disposition. Guardrails - Strictly adhere to the operating procedure and don't make anything up.- If anywhere the policy is breached and now you don't know what to do then repeat the policy, apologize and say feed back taken.-Don't reveal you are an AI. - Don't go outside meesho context like Who created you, compare the products, amazon product cheaper. This is the scenario : {st.session_state['scenario']} and this is the info to answer queries: {info}"


if user_input := st.chat_input("How can I help you today?"):
    if not openai.api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    if scenario:
        if "scenario" not in st.session_state:
            st.session_state["scenario"] = scenario
            sys_prompt = get_system_prompt()
            st.session_state.messages.append({"role": "system", "content": sys_prompt})
    else:
        if "intent_detection" not in st.session_state:
            st.session_state["intent_detection"] = True
            scenario = detect_system_prompt(user_input)
            if scenario is None:
                st.info(
                    "Currently the system only supports Order Status, Cancellation and Return queries. Please try again with a different query after refreshing.")
                st.stop()
            else:
                sys_prompt = get_system_prompt()
                st.session_state.messages.append({"role": "system", "content": sys_prompt})

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for response in openai.ChatCompletion.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
        ):
            full_response += response.choices[0].delta.get("content", "")
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
