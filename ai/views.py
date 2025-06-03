from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .tasks import retrain_prophet_model


class TriggerProphetTrainingView(APIView):
    def post(self, request):
        print('training data ', request.data)
        try:
            customer_id = request.data["customer_id"]
            sales_data = request.data["sales_data"]
            
            if not isinstance(sales_data, list) or len(sales_data) < 3:
                return Response({"error": "Insufficient sales data"}, status=400)

            retrain_prophet_model.delay(customer_id, sales_data)
            return Response({"status": "Task triggered"}, status=202)

        except KeyError as e:
            return Response({"error": f"Missing field: {str(e)}"}, status=400)
        except Exception as e:
            return Response({"error": str(e)}, status=500)
