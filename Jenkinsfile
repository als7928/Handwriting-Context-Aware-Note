pipeline {
    // 반드시 관리자가 지정한 'docker' 라벨 사용
    agent { label 'docker' } 

    tools {
        // 별칭 대신 전체 클래스 경로를 사용합니다. 
        // 'jenkins-docker'는 Global Tool Configuration에 등록한 Name과 반드시 같아야 합니다.
        'org.jenkinsci.plugins.docker.commons.tools.DockerTool' 'jenkins-docker'
    }

    environment {
        HARBOR_URL = 'amdp-registry.skala-ai.com'
        HARBOR_PROJECT = 'skala26a-ai2'
        HARBOR_CREDS = 'harbor-robot-account' 
        
        BACKEND_IMAGE = 'sk047-myservice-backend'
        BACKEND_VER = '1.0.4'
        FRONTEND_IMAGE = 'sk047-myservice-frontend'
        FRONTEND_VER = '1.0.1'
    }

    stages {
        stage('Checkout') {
            steps {
                echo '>>> Stage 1: Checkout'
                checkout scm
            }
        }

        stage('Build') {
            steps {
                script {
                    echo '>>> Stage 2: Build'
                    // 도구가 올바르게 로드되었다면 sh "docker ..."가 실행됩니다.
                    sh "docker build -t ${HARBOR_URL}/${HARBOR_PROJECT}/${BACKEND_IMAGE}:${BACKEND_VER} -f backend/Dockerfile-backend ./backend"
                    sh "docker build -t ${HARBOR_URL}/${HARBOR_PROJECT}/${FRONTEND_IMAGE}:${FRONTEND_VER} -f frontend/Dockerfile-frontend ./frontend"
                }
            }
        }

        stage('Test') {
            steps {
                echo '>>> Stage 3: Test'
                sh "docker --version"
            }
        }

        stage('Deploy') {
            steps {
                script {
                    echo '>>> Stage 4: Deploy'
                    // Harbor 인증 및 이미지 푸시
                    docker.withRegistry("https://${HARBOR_URL}", "${HARBOR_CREDS}") {
                        sh "docker push ${HARBOR_URL}/${HARBOR_PROJECT}/${BACKEND_IMAGE}:${BACKEND_VER}"
                        sh "docker push ${HARBOR_URL}/${HARBOR_PROJECT}/${FRONTEND_IMAGE}:${FRONTEND_VER}"
                    }
                }
            }
        }
    }

    post {
        success {
            echo 'SUCCESS: Both backend and frontend images are pushed.'
            sh "docker rmi ${HARBOR_URL}/${HARBOR_PROJECT}/${BACKEND_IMAGE}:${BACKEND_VER} || true"
            sh "docker rmi ${HARBOR_URL}/${HARBOR_PROJECT}/${FRONTEND_IMAGE}:${FRONTEND_VER} || true"
        }
        failure {
            echo 'FAILURE: If "docker: not found" persists, check Global Tool Configuration Name.'
        }
    }
}