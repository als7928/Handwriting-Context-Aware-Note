pipeline {
    agent any

    stages {
        stage('Checkout') {
            steps {
                echo 'GitHub에서 소스코드를 가져옵니다.'
                checkout scm
            }
        }
        stage('Build') {
            steps {
                echo '빌드(설치) 과정을 수행합니다.'
                // 예: npm install 또는 pip install
                sh 'echo "Installing dependencies..."'
            }
        }
        stage('Test') {
            steps {
                echo '테스트를 수행합니다.'
                sh 'echo "Running tests..."'
            }
        }
        stage('Deploy') {
            steps {
                echo '배포를 수행합니다.'
                sh 'echo "Deploying application..."'
            }
        }
    }
}