pipeline {
    // 반드시 관리자에게 확인받은 'docker' 라벨을 사용하세요.
    agent { label 'docker' } 

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
                    
                    // Global Tool Configuration에서 만든 'Name'과 반드시 일치해야 합니다.
                    def dockerHome = tool name: 'jenkins-docker', type: 'dockerTool'
                    
                    // 설치된 도구의 bin 폴더를 PATH 맨 앞에 추가하여 실행합니다.
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
                        // Harbor 로그인 및 푸시 (Docker Pipeline 플러그인 활용)
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
            // 로컬 이미지 정리
            script {
                try {
                    def dockerHome = tool name: 'jenkins-docker', type: 'dockerTool'
                    withEnv(["PATH+DOCKER=${dockerHome}/bin"]) {
                        sh "docker rmi ${HARBOR_URL}/${HARBOR_PROJECT}/${BACKEND_IMAGE}:${BACKEND_VER} || true"
                        sh "docker rmi ${HARBOR_URL}/${HARBOR_PROJECT}/${FRONTEND_IMAGE}:${FRONTEND_VER} || true"
                    }
                } catch (e) {
                    echo "Cleanup skipped: ${e.message}"
                }
            }
        }
        failure {
            echo 'FAILURE: Check if the Tool Name "jenkins-docker" is correct in Global Tool Configuration.'
        }
    }
}