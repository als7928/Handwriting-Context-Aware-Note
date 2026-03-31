pipeline {
    // 1. Agent label: docker
    agent { label 'docker' } 

    // 2. Tool type을 'docker'에서 'dockerTool'로 변경합니다.
    // 'jenkins-docker'는 Global Tool Configuration에 등록하신 Name과 같아야 합니다.
    tools {
        dockerTool 'jenkins-docker' 
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
                    // Global Tool Configuration에서 만든 이름을 변수에 담음
                    def dockerBin = tool name: 'jenkins-docker', type: 'dockerTool'
                    
                    // Docker 실행 파일의 경로를 환경변수에 강제로 추가
                    withEnv(["PATH+DOCKER=${dockerBin}/bin"]) {
                        sh "docker build -t ${HARBOR_URL}/${HARBOR_PROJECT}/${BACKEND_IMAGE}:${BACKEND_VER} -f backend/Dockerfile-backend ./backend"
                        sh "docker build -t ${HARBOR_URL}/${HARBOR_PROJECT}/${FRONTEND_IMAGE}:${FRONTEND_VER} -f frontend/Dockerfile-frontend ./frontend"
                    }
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
            echo 'SUCCESS: All stages finished.'
            sh "docker rmi ${HARBOR_URL}/${HARBOR_PROJECT}/${BACKEND_IMAGE}:${BACKEND_VER} || true"
            sh "docker rmi ${HARBOR_URL}/${HARBOR_PROJECT}/${FRONTEND_IMAGE}:${FRONTEND_VER} || true"
        }
    }
}