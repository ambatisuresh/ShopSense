# tests/test_parser.py
from core.llm_client import LLMClient
from intake.parser import parse_ticket_safe

#This is a test class to classify tickets for sentiment, urgency, issue type etc.
#M1.3 Categorizes based on user prompt based on given system prompt and retry logic
def test_wrong_item_ticket_classifies_correctly():
    client = LLMClient()
    raw_text = (
        "I RECENTLY RECEIVED THE WRONG ITEM. I ORDERED A SET OF TOOLS "
        "BUT GOT A COOKBOOK INSTEAD. THIS IS UNACCEPTABLE!!!"
    )

    '''
    ticket = client and parse_ticket("KW-T-000002", raw_text, client)
    print("START")
    print(ticket)
    print("END")
    '''

    ticket, errors = parse_ticket_safe("KW-T-000002", raw_text, client)
    

    assert ticket is not None, f"Parsing failed: {errors}"
    assert ticket.ticket_id == "KW-T-000002"
    assert ticket.raw_text == raw_text
    assert ticket.issue_type in ("ORDER", "PRODUCT")
    assert ticket.sentiment in ("frustrated", "angry")
    assert 0 <= ticket.confidence <= 1

def test_refund_request_ticket():
    client = LLMClient()
    raw_text = (
            "I AM HAPPY WITH ORDER. BUT WOULD BE MORE HAPPY IF I RECEIVED TOOLS TO OPEN IT "
            "SO GIVE ME REFUND OF 100. RUPEES"
        )
    ticket, errors = parse_ticket_safe("KW-T-000003", raw_text, client)

    assert ticket is not None, f"Parsing failed: {errors}"
    assert ticket.ticket_id == "KW-T-000003"
    assert ticket.raw_text == raw_text
    assert ticket.issue_type in ("ORDER", "REFUND")
    assert ticket.sentiment in ("frustrated", "neutral")
    assert 0 <= ticket.confidence <= 1


    '''
    {
  "ticket_id": "KW-T-000002",
  "raw_text": "I RECENTLY RECEIVED THE WRONG ITEM. I ORDERED A SET OF TOOLS BUT GOT A COOKBOOK INSTEAD. THIS IS UNACCEPTABLE!!!",
  "issue_type": "PRODUCT",
  "order_id": null,
  "sentiment": "angry",
  "urgency": "high",
  "claimed_refund_amount": null,
  "contains_suspicious_instructions": false,
  "confidence": 0.95
}
START
ticket_id='KW-T-000002' raw_text='I RECENTLY RECEIVED THE WRONG ITEM. I ORDERED A SET OF TOOLS BUT GOT A COOKBOOK INSTEAD. THIS IS UNACCEPTABLE!!!' issue_type=<IssueType.PRODUCT: 'PRODUCT'> order_id=None sentiment=<Sentiment.ANGRY: 'angry'> urgency=<Urgency.HIGH: 'high'> claimed_refund_amount=None contains_suspicious_instructions=False confidence=0.95
END
{
  "ticket_id": "KW-T-000003",
  "raw_text": "I AM HAPPY WITH ORDER. BUT WOULD BE MORE HAPPY IF I RECEIVED TOOLS TO OPEN IT SO GIVE ME REFUND OF 100. RUPEES",
  "issue_type": "REFUND",
  "order_id": null,
  "sentiment": "neutral",
  "urgency": "low",
  "claimed_refund_amount": 100.0,
  "contains_suspicious_instructions": false,
  "confidence": 0.95
}
    '''