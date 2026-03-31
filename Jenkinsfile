pipeline {
    agent { label 'docker' } 

    tools {
        // 클래스 경로와 이름을 정확히 명시
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
                checkout scm
            }
        }

        stage('Build') {
            steps {
                script {
                    echo '>>> Stage 2: Build with Latest Docker'
                    
                    // 도구 경로를 다시 한번 확인하고 PATH 최상단에 주입
                    def dockerHome = tool name: 'jenkins-docker', type: 'org.jenkinsci.plugins.docker.commons.tools.DockerTool'
                    
                    withEnv(["PATH+DOCKER=${dockerHome}/bin"]) {
                        // 현재 사용 중인 도커 버전을 로그로 출력해서 확인 (1.29인지 최신인지)
                        sh "docker version" 
                        
                        sh "docker build -t ${HARBOR_URL}/${HARBOR_PROJECT}/${BACKEND_IMAGE}:${BACKEND_VER} -f backend/Dockerfile-backend ./backend"
                        sh "docker build -t ${HARBOR_URL}/${HARBOR_PROJECT}/${FRONTEND_IMAGE}:${FRONTEND_VER} -f frontend/Dockerfile-frontend ./frontend"
                    }
                }
            }
        }

        stage('Deploy') {
            steps {
                script {
                    echo '>>> Stage 4: Deploy'
                    def dockerHome = tool name: 'jenkins-docker', type: 'org.jenkinsci.plugins.docker.commons.tools.DockerTool'
                    withEnv(["PATH+DOCKER=${dockerHome}/bin"]) {
                        docker.withRegistry("https://${HARBOR_URL}", "${HARBOR_CREDS}") {
                            sh "docker push ${HARBOR_URL}/${HARBOR_PROJECT}/${BACKEND_IMAGE}:${BACKEND_VER}"
                            sh "docker push ${HARBOR_URL}/${HARBOR_PROJECT}/${FRONTEND_IMAGE}:${FRONTEND_VER}"
                        }
                    }
                }
            }
        }
    }
}