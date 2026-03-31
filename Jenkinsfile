pipeline {
    agent { label 'docker' } 

    tools {
        // Global Tool Configuration에서 등록한 이름과 일치해야 합니다.
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
                    
                    // 1. 도구가 설치된 경로를 직접 변수에 담습니다.
                    def dockerHome = tool name: 'jenkins-docker', type: 'dockerTool'
                    
                    // 2. 해당 경로의 bin 폴더를 PATH에 추가하여 실행합니다.
                    withEnv(["PATH+DOCKER=${dockerHome}/bin"]) {
                        echo "Using Docker from: ${dockerHome}/bin"
                        
                        // Backend Build
                        sh "docker build -t ${HARBOR_URL}/${HARBOR_PROJECT}/${BACKEND_IMAGE}:${BACKEND_VER} -f backend/Dockerfile-backend ./backend"
                        
                        // Frontend Build
                        sh "docker build -t ${HARBOR_URL}/${HARBOR_PROJECT}/${FRONTEND_IMAGE}:${FRONTEND_VER} -f frontend/Dockerfile-frontend ./frontend"
                    }
                }
            }
        }

        stage('Test') {
            steps {
                echo '>>> Stage 3: Test'
                script {
                    def dockerHome = tool name: 'jenkins-docker', type: 'dockerTool'
                    withEnv(["PATH+DOCKER=${dockerHome}/bin"]) {
                        sh "docker --version"
                    }
                }
            }
        }

        stage('Deploy') {
            steps {
                script {
                    echo '>>> Stage 4: Deploy'
                    def dockerHome = tool name: 'jenkins-docker', type: 'dockerTool'
                    
                    withEnv(["PATH+DOCKER=${dockerHome}/bin"]) {
                        // docker.withRegistry 구문도 내부적으로 docker 명령어를 쓰므로 PATH 안에서 실행
                        docker.withRegistry("https://${HARBOR_URL}", "${HARBOR_CREDS}") {
                            sh "docker push ${HARBOR_URL}/${HARBOR_PROJECT}/${BACKEND_IMAGE}:${BACKEND_VER}"
                            sh "docker push ${HARBOR_URL}/${HARBOR_PROJECT}/${FRONTEND_IMAGE}:${FRONTEND_VER}"
                        }
                    }
                }
            }
        }
    }

    post {
        success {
            echo 'SUCCESS: All images pushed to Harbor.'
        }
        failure {
            echo 'FAILURE: Still getting "docker: not found". Please check Global Tool Configuration.'
        }
    }
}