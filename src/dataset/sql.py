from typing import Annotated,TypedDict, List, Optional,Any, Union, Mapping, cast 
from pydantic import BaseModel
from sqlmodel import select, Session, SQLModel, create_engine, Field
from settings.types import VietnamLaw
from pandas import DataFrame, Series, isna
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage

class VietnamLawModel(SQLModel, table=True):
    id: int = Field(primary_key=True)
    document_number: Optional[str]
    content: str
    title: Optional[str]
    url: Optional[str]
    legal_type: Optional[str]
    legal_sectors: Optional[str]
    issuing_authority: Optional[str]
    issuance_date: Optional[str]
    signers: Optional[str]

class LogEntryModel(SQLModel, table=True):
    id: int = Field(primary_key=True)
    timestamp: str
    user_message: str
    bot_response: str

class ChatroomModel(SQLModel, table = True):
    roomId: str = Field(primary_key=True)
    name: Optional[str] 
    messages: List[str]

    @property
    def langchain_messages(self) -> List[AnyMessage]:
        list_messages = []
        for message_str in self.messages:
            try:
                message_dict = eval(message_str)
                if isinstance(message_dict, dict) and "type" in message_dict and "content" in message_dict:
                    content = message_dict["content"]
                    if message_dict["type"] == "human":
                        list_messages.append(HumanMessage(content=content))
                    elif message_dict["type"] == "ai":
                        list_messages.append(AIMessage(content=content))
            except Exception as exc:
                print(f"Error parsing message: {message_str}, error: {exc}")
                continue
        return list_messages
    
SQLModelArgument = Union[SQLModel, Mapping[str,Any]]

class SQLiteDatabase:
    path : str 
    name : str 
    model : type[SQLModel] = VietnamLawModel
    engine : Any = None

    @staticmethod
    def normalize_record(data: Mapping[str, Any] | Series) -> dict[str, Any]:
        if isinstance(data, Series):
            data = cast(dict[str, Any], data.to_dict())

        normalized_data = {}
        for key, value in dict(data).items():
            try:
                normalized_data[key] = None if isna(value) else value
            except TypeError:
                normalized_data[key] = value

        return normalized_data

    @staticmethod
    def build_record(data, model : type[SQLModel]) -> SQLModel:
        if isinstance(data, model):
            return data
        if isinstance(data, (Mapping, Series)):
            data = SQLiteDatabase.normalize_record(data)
        return model.model_validate(data)
    
    def __init__(self, path: str, name: str, model: Optional[type[SQLModel]] = None):
        self.path = path
        self.name = name
        if model:
            self.model = model
        url = f"sqlite:///{self.path}/{self.name}.db"
        self.engine = create_engine(url)
        SQLModel.metadata.create_all(self.engine)

    def create_database(self):
        SQLModel.metadata.create_all(self.engine)

    def create_table(self):
        SQLModel.metadata.create_all(self.engine, tables=[self.model.__table__])

    def drop_table(self):
        SQLModel.metadata.drop_all(self.engine, tables=[self.model.__table__])

    def all(self) -> List[SQLModel]:
        with Session(self.engine) as session:
            statement = select(self.model)
            results = session.exec(statement).all()
            return [self.model.model_validate(result.model_dump()) for result in results]

    def add(self, law: SQLModelArgument):
        with Session(self.engine) as session:
            law_model = self.build_record(law, self.model)
            session.add(law_model)
            session.commit()

    def add_many(self, laws: List[SQLModelArgument] | DataFrame):
        with Session(self.engine) as session:
            if isinstance(laws, DataFrame):
                laws = [self.build_record(law, self.model) for _, law in laws.iterrows()]
            else:
                laws = [self.build_record(law, self.model) for law in laws]
            session.add_all(laws)
            session.commit()

    def delete(self, id):
        with Session(self.engine) as session:
            law = session.get(self.model, id)
            if law:
                session.delete(law)
                session.commit()
    
    def get(self, id) -> Optional[SQLModel]:
        with Session(self.engine) as session:
            law_model = session.get(self.model, id)
            if law_model:
                return self.model.model_validate(law_model.model_dump())
            return None

    def get_at_index(self, index) -> Optional[SQLModel]:
        with Session(self.engine) as session:
            statement = select(self.model).offset(index).limit(1)
            result = session.exec(statement).first()
            if result:
                return self.model.model_validate(result.model_dump())
            return None

    def update(self, id, updated_fields: dict):
        with Session(self.engine) as session:
            law = session.get(self.model, id)
            if law:
                for key, value in updated_fields.items():
                    setattr(law, key, value)
                session.add(law)
                session.commit()

    def have_record(self, id) -> bool:
        with Session(self.engine) as session:
            law = session.get(self.model, id)
            return law is not None
          
    