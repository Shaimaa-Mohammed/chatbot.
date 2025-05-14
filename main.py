import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory


user_lands= {} 

#Set the LLM
load_dotenv()  
api_key = os.getenv("GOOGLE_API_KEY")
llm= ChatGoogleGenerativeAI(model='gemini-2.0-flash-001')

# llm= get_api()

#Set Prompt
soil_analysis_prompt = PromptTemplate(
    input_variables=[
        "history", "input",
        "soil_ph", "soil_moisture", "soil_air", "soil_temp", "ambient_temp",
        "humidity", "light_intensity", "nitrogen_level", "potassium_level", "phosphorus_level",
        "chlorophyll_content", "electrochemical_signal", "organic_matter", "soil_type", "disease", "plant_status"
    ],
    template="""
    أنت خبير في تحليل التربة وصحة النباتات.

    المحادثة السابقة:
    {history}

    بيانات التربة والنبات:
    1. pH التربة: {soil_ph}
    2. رطوبة التربة: {soil_moisture}%
    3. هواء التربة: {soil_air}
    4. درجة حرارة التربة: {soil_temp}°C
    5. درجة الحرارة المحيطة: {ambient_temp}°C
    6. الرطوبة في الهواء: {humidity}%
    7. شدة الإضاءة: {light_intensity} Lux
    8. النيتروجين: {nitrogen_level} mg/kg
    9. البوتاسيوم: {potassium_level} mg/kg
    10. الفسفور: {phosphorus_level} mg/kg
    11. الكلوروفيل: {chlorophyll_content} mg/m²
    12. الإشارة الكهربائية الكيميائية: {electrochemical_signal} mV
    13. المادة العضوية: {organic_matter}%
    14. نوع التربة: {soil_type}
    15. المرض الظاهر: {disease}
    16. حالة النبات: {plant_status}

    الطلب الحالي:
    {input}

    الرجاء الرد بناءً على البيانات أعلاه وسياق المحادثة.
    نسق الإجابة بشكل نقاط واضحة أو فقرات مرتبة ومختصره لتادية غرض السوال فقط ومبسطة ليسهل على المزارع الفهم والتطبيق.
"""
)

plant_disease_prompt = PromptTemplate(
    input_variables=["disease_name"],
    template="""
أنت مساعد زراعي محترف ومتخصص في أمراض النباتات وطرق علاجها والوقاية منها.

قام المستخدم برفع صورة لنبتة، وتم تشخيص إصابتها بالمرض التالي: {disease_name}.

من فضلك قدّم للمستخدم ما يلي:
1. شرح مبسط وواضح عن هذا المرض وكيف يؤثر على النباتات بشكل عام.
2. خطوات دقيقة وعملية لعلاج هذا المرض.
3. نصائح فعالة للوقاية من هذا المرض في المستقبل.
4. إذا أمكن، اقترح طرق علاج عضوية أو صديقة للبيئة.

استخدم أسلوبًا سهلًا ومطمئنًا، وتجنب المصطلحات المعقدة، مع التركيز على تقديم تعليمات يمكن تنفيذها بسهولة من قِبل المستخدم العادي.
"""
)

app= FastAPI()

class Message(BaseModel):
    user_input: str
    soil_ph: Optional[float]= None
    soil_moisture: Optional[float]= None
    soil_air: Optional[float]= None
    soil_temp: Optional[float]= None
    ambient_temp: Optional[float]= None
    humidity: Optional[float]= None
    light_intensity: Optional[float]= None
    nitrogen_level: Optional[float]= None
    potassium_level: Optional[float]= None
    phosphorus_level: Optional[float]= None
    chlorophyll_content: Optional[float]= None
    electrochemical_signal: Optional[float]= None
    organic_matter: Optional[float]= None
    soil_type: Optional[str]= None
    disease: Optional[str]= None
    plant_status: Optional[str]= None

class Disease(BaseModel):
    disease_name: str


chain_template= LLMChain(llm=llm, prompt=soil_analysis_prompt, memory=None, verbose=True)
advice_chain= LLMChain(llm=llm , prompt=plant_disease_prompt)


@app.post('/start-chat/{user_name}/{land_name}')
def start_chat(user_name: str, land_name: str, message: Message):
    if land_name not in user_lands:
        user_lands[land_name]= {
            "memory": ConversationBufferMemory(memory_key="history", input_key="input", return_messages=True),
            "soil_data": {} 
        }

    memory= user_lands[land_name]["memory"]
    soil_data= user_lands[land_name]["soil_data"]

    new_soil_data= {
        "soil_ph": message.soil_ph or soil_data.get("soil_ph"),
        "soil_moisture": message.soil_moisture or soil_data.get("soil_moisture"),
        "soil_air": message.soil_air or soil_data.get("soil_air"),
        "soil_temp": message.soil_temp or soil_data.get("soil_temp"),
        "ambient_temp": message.ambient_temp or soil_data.get("ambient_temp"),
        "humidity": message.humidity or soil_data.get("humidity"),
        "light_intensity": message.light_intensity or soil_data.get("light_intensity"),
        "nitrogen_level": message.nitrogen_level or soil_data.get("nitrogen_level"),
        "potassium_level": message.potassium_level or soil_data.get("potassium_level"),
        "phosphorus_level": message.phosphorus_level or soil_data.get("phosphorus_level"),
        "chlorophyll_content": message.chlorophyll_content or soil_data.get("chlorophyll_content"),
        "electrochemical_signal": message.electrochemical_signal or soil_data.get("electrochemical_signal"),
        "organic_matter": message.organic_matter or soil_data.get("organic_matter"),
        "soil_type": message.soil_type or soil_data.get("soil_type"),
        "disease": message.disease or soil_data.get("disease"),
        "plant_status": message.plant_status or soil_data.get("plant_status")
    }


    user_lands[land_name]["soil_data"] = new_soil_data

    inputs= {
        "history": memory.buffer,
        "input": message.user_input,
        **new_soil_data
    }

    chain = chain_template.copy()
    chain.memory= memory

    llm_response= chain.run(inputs)
    formatted_response = llm_response.replace("\n", "<br>").replace("**", "<b>").replace("<b>", "</b>", 1)

    return {"Response": formatted_response}


@app.post('/give-me-advice/{user_name}')
def give_me_advice(user_name: str , disease: Disease):

    input= {"disease_name": disease.disease_name}
    llm_response= advice_chain.run(input)
    formatted_response = llm_response.replace("\n", "<br>").replace("**", "<b>").replace("<b>", "</b>", 1)

    return {"Response": formatted_response}



