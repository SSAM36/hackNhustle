#!/bin/bash
# Railway startup script

echo "Starting ISL Learning Platform Backend..."
echo "Environment: $ENVIRONMENT"
echo "Port: $PORT"

# Check if MongoDB URI is set
if [ -z "$MONGO_URI" ]; then
    echo "WARNING: MONGO_URI environment variable is not set!"
    echo "Some features will not work without a database connection."
fi

# Check if JWT_SECRET_KEY is set
if [ -z "$JWT_SECRET_KEY" ]; then
    echo "WARNING: JWT_SECRET_KEY not set! Using fallback (insecure for production)"
fi

# Start the application with gunicorn
exec gunicorn app:app --bind 0.0.0.0:$PORT --workers 4 --timeout 120 --access-logfile - --error-logfile -
